from hera import EnvSpec, ImagePullPolicy
from hera.artifact import InputArtifact, S3Artifact, OutputArtifact
from hera.resources import Resources
from hera.task import Task
from hera.toleration import GPUToleration
from hera.workflow import Workflow
from hera.workflow_service import WorkflowService
from hera.variable import VariableAsEnv

from cloudevents.sdk.event import v1
from dapr.clients import DaprClient
from dapr.ext.grpc import App


def pixray_upload(id: str):
  import json
  from minio import Minio
  import os
  import glob

  def upload_local_directory_to_minio(local_path, bucket_name, minio_path):
    assert os.path.isdir(local_path)
    client = Minio("dumb.dev", access_key=os.getenv('NIGHTMAREBOT_MINIO_KEY'), secret_key=os.getenv('NIGHTMAREBOT_MINIO_SECRET'))

    for local_file in glob.glob(local_path + '/**'):
        local_file = local_file.replace(os.sep, "/") # Replace \ with / on Windows
        if not os.path.isfile(local_file):
            upload_local_directory_to_minio(
                local_file, bucket_name, minio_path + "/" + os.path.basename(local_file))
        else:
            content_type = "application/octet-stream"
            if local_file.endswith("png"):
              content_type = "image/png"
            if local_file.endswith("mp4"):
              content_type = "video/mp4"
            if local_file.endswith("jpg"):
              content_type = "image/jpg"
            remote_path = os.path.join(
                minio_path, local_file[1 + len(local_path):])
            remote_path = remote_path.replace(
                os.sep, "/")  # Replace \ with / on Windows
            client.fput_object(bucket_name, remote_path, local_file, content_type=content_type)

  upload_local_directory_to_minio('/result', "nightmarebot-output", id)


def upload_swinir(id: str):
  from minio import Minio
  import os
  client = Minio("dumb.dev", access_key=os.getenv('NIGHTMAREBOT_MINIO_KEY'), secret_key=os.getenv('NIGHTMAREBOT_MINIO_SECRET'))
  client.fput_object("nightmarebot-output", f'{id}/output.png', '/result/input_SwinIR.png', content_type="image/png")

def pixray_prepare(id: str):
  from minio import Minio
  import yaml
  import sys
  import os
  import requests
  client = Minio("dumb.dev", access_key=os.getenv('NIGHTMAREBOT_MINIO_KEY'), secret_key=os.getenv('NIGHTMAREBOT_MINIO_SECRET'))
  os.makedirs('/tmp/pixray')
  outfile = os.path.join('/tmp/pixray', 'input.yaml')
  configfile = os.path.join('/tmp/pixray', 'config.yaml')
  contextfile = os.path.join('/tmp/pixray', 'context.json')
  client.fget_object('nightmarebot-workflow', f'{id}/input.yaml', outfile)
  client.fget_object('nightmarebot-workflow', f'{id}/context.json', contextfile)
  client.fget_object('nightmarebot-workflow', f'{id}/prompt.txt', '/tmp/pixray/prompt.txt')
  client.fget_object('nightmarebot-workflow', f'{id}/id.txt', '/tmp/pixray/id.txt')
  with open (outfile) as stream:
    try:
      request_settings = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
      print("Problem with settings", exc)
      sys.exit(1)
  try:
    if request_settings["init_image"] and not str(request_settings["init_image"]).isspace():
      outfile = os.path.join('/tmp/pixray', 'init_image.png')
      with requests.get(request_settings["init_image"], stream = True) as r:
        with open(outfile, 'wb') as f:
          for chunk in r.iter_content(chunk_size = 16*1024):
            f.write(chunk)
      request_settings["init_image"] = outfile
  except Exception as e: print(f'Error saving init_image:{e}', flush=True)

  with open (configfile, 'w') as outstream:
    yaml.safe_dump(request_settings, outstream)

def enhance_prepare(id: str):
  from minio import Minio
  import os,sys
  client = Minio("dumb.dev", access_key=os.getenv('NIGHTMAREBOT_MINIO_KEY'), secret_key=os.getenv('NIGHTMAREBOT_MINIO_SECRET'))
  os.makedirs('/tmp/enhance/lq')
  client.fget_object('nightmarebot-workflow', f'{id}/input.png', '/tmp/enhance/lq/input.png')
  client.fget_object('nightmarebot-workflow', f'{id}/context.json', '/tmp/enhance/context.json')
  client.fget_object('nightmarebot-workflow', f'{id}/prompt.txt', '/tmp/enhance/prompt.txt')
  client.fget_object('nightmarebot-workflow', f'{id}/id.txt', '/tmp/enhance/id.txt')

app = App()

@app.subscribe(pubsub_name="jetstream-pubsub", topic='request.swinir')
def enhance(event: v1.Event) -> None:
  import os
  import json

  data = json.loads(event.Data())
  id = data["id"]

  try:
    ws = WorkflowService(host='https://argo-server.argo:2746', token='', verify_ssl=False, namespace='argo')
    w = Workflow(f'enhance-{id}', ws, parallelism=1, namespace='argo')
    gke_k80_gpu = {'cloud.google.com/gke-accelerator': 'nvidia-tesla-k80'}
    gke_t4_gpu = {'cloud.google.com/gke-accelerator': 'nvidia-tesla-t4'}

    command=[
      'python3',
      'main_test_swinir.py',
      '--task',
      'real_sr',
      '--scale',
      '4',
      '--large_model',
      '--model_path',
      'model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth',
      '--folder_lq',
      '/tmp/enhance/lq'
    ]

    minio_key = str(os.getenv('NIGHTMAREBOT_MINIO_KEY'))
    minio_secret = str(os.getenv('NIGHTMAREBOT_MINIO_SECRET'))
    bot_token = str(os.getenv('NIGHTMAREBOT_TOKEN'))

    p_t = Task(
      'swinir-prepare',
      enhance_prepare,
      [{'id': id}],
      image_pull_policy=ImagePullPolicy.IfNotPresent,
      image='us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest',
      env_specs=[EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key), EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret)],
      output_artifacts=[OutputArtifact(name='input', path='/tmp/enhance')]
    )

    e_t = Task(
      'swinir-enhance',
      image='us-docker.pkg.dev/nightmarebot-ai/nightmarebot/swinir-enhance:latest',
      image_pull_policy=ImagePullPolicy.IfNotPresent,
      command=command,
      resources=Resources(gpus=1),
      tolerations=[GPUToleration],
#      node_selectors=gke_t4_gpu,
      input_artifacts=[InputArtifact(name='input', path='/tmp/enhance', from_task='swinir-prepare', artifact_name='input')],
      output_artifacts=[OutputArtifact(name='result', path='/src/results/swinir_real_sr_x4_large/')]
    )

    u_t = Task(
      'swinir-upload',
      upload_swinir,
      [{'id': id}],
      image_pull_policy=ImagePullPolicy.IfNotPresent,
      image='us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest',
      env_specs=[EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key), EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret), EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token)],
      input_artifacts=[InputArtifact(name='result', path='/result', from_task='swinir-enhance', artifact_name='result')]
    )

    r_t = Task(
      'swinir-respond',
      image='us-docker.pkg.dev/nightmarebot-ai/nightmarebot/nightmarebot-publish:latest',
      image_pull_policy=ImagePullPolicy.IfNotPresent,
      command=['dotnet', 'NightmareBot.Publish.dll'],
      env_specs=[EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key), EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret), EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token)],
      input_artifacts=[InputArtifact(name='input', path='/result/swinir', from_task='swinir-prepare', artifact_name='input'),
        InputArtifact(name='result', path='/result/swinir/output', from_task='swinir-enhance', artifact_name='result')]
    )

    p_t >> e_t >> u_t >> r_t
    w.add_tasks(p_t, e_t, u_t, r_t)
    w.create()
  except Exception as e: print(f'Error enqueing request:{e}', flush=True)


@app.subscribe(pubsub_name="jetstream-pubsub", topic='request.pixray')
def dream(event: v1.Event) -> None:
  import os
  import json

  data = json.loads(event.Data())
  id = data["id"]

  try:
    ws = WorkflowService(host='https://argo-server.argo:2746', token='', verify_ssl=False, namespace='argo')  
    w = Workflow(f'pixray-{id}', ws, parallelism=1, namespace='argo')
    gke_t4_gpu = {'cloud.google.com/gke-accelerator': 'nvidia-tesla-t4'}

    command=[
        'python',
        'draw.py',
        '/tmp/pixray'
    ]

    minio_key = str(os.getenv('NIGHTMAREBOT_MINIO_KEY'))
    minio_secret = str(os.getenv('NIGHTMAREBOT_MINIO_SECRET'))
    bot_token = str(os.getenv('NIGHTMAREBOT_TOKEN'))

    p_t = Task(
      'pixray-prepare',
      pixray_prepare,
      [{'id': id}],
      image='us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest',
      env_specs=[EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key), EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret)],
      output_artifacts=[OutputArtifact(name='input', path='/tmp/pixray')]
    )

    d_t = Task(
      'pixray-dreamer', 
      image='us-docker.pkg.dev/nightmarebot-ai/nightmarebot/pixray-dreamer:latest', 
      image_pull_policy=ImagePullPolicy.IfNotPresent,
      command=command,
      resources=Resources(gpus=1),
      tolerations=[GPUToleration],
      #node_selectors=gke_t4_gpu,
      env_specs=[EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key), EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret)],
      input_artifacts=[InputArtifact(name='input', path='/tmp/pixray', from_task='pixray-prepare', artifact_name='input')],
      output_artifacts=[OutputArtifact(name='result', path='/tmp/pixray')]
    )

    u_t = Task(
      'pixray-upload',
      pixray_upload, 
      [{'id': id}],
      image='us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest',    
      env_specs=[EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key), EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret), EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token)],
      input_artifacts=[InputArtifact(name='result', path='/result', from_task='pixray-dreamer', artifact_name='result')]
    )

    r_t = Task(
      'pixray-respond',
      image_pull_policy=ImagePullPolicy.IfNotPresent,
      image='us-docker.pkg.dev/nightmarebot-ai/nightmarebot/nightmarebot-publish:latest',
      command=['dotnet', 'NightmareBot.Publish.dll'],
      env_specs=[EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key), EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret), EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token)],
      input_artifacts=[InputArtifact(name='result', path='/result/pixray', from_task='pixray-dreamer', artifact_name='result')]
    )

    p_t >> d_t >> u_t >> r_t
    w.add_tasks(p_t, d_t, u_t, r_t)
    print(f'creating workflow for prompt {data["input"]["prompts"]}',flush=True)
    w.create()
  except Exception as e: print(f'Error enqueing request:{e}', flush=True)

app.run(50055)
