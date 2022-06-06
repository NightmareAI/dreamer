from unicodedata import name
from hera import EnvSpec, ImagePullPolicy
from hera.artifact import InputArtifact, S3Artifact, OutputArtifact
from hera.resources import Resources
from hera.task import Task
from hera.toleration import GPUToleration
from hera.retry import Retry
from hera.workflow import Workflow
from hera.workflow_service import WorkflowService
from hera.variable import VariableAsEnv
from hera.volumes import ExistingVolume

from cloudevents.sdk.event import v1
from dapr.clients import DaprClient
from dapr.ext.grpc import App


publish_image = "us-central1-docker.pkg.dev/nightmarebot-ai/nightmarebot/nightmarebot-publish@sha256:90b7b44b00ff5d4955d02e7b217b5a503948010d35a89616911870e856382aeb"
majesty_image = "us-central1-docker.pkg.dev/nightmarebot-ai/nightmarebot/majesty-diffusion@sha256:dd2a925a27e91b294c8a5f6cfc285b5dd8d296259f9852a153c9f3819714766c"
latent_diffusion_image = "us-central1-docker.pkg.dev/nightmarebot-ai/nightmarebot/latent-diffusion-dreamer@sha256:0546cc68f4f0eeea8af386aaa2b31e73b3c7dfc3f56ea8fd33c474b5b1cc6939"
esrgan_image = "us-central1-docker.pkg.dev/nightmarebot-ai/nightmarebot/esrgan-enhance@sha256:e99a97d83fd49154948d9455d1226fdc87687c9e7d2b065e2318633374043281"


def result_upload(id: str):
    import json
    from minio import Minio
    import os
    import glob

    client = Minio(
        "dumb.dev",
        access_key=os.getenv("NIGHTMAREBOT_MINIO_KEY"),
        secret_key=os.getenv("NIGHTMAREBOT_MINIO_SECRET"),
    )

    def upload_local_directory_to_minio(local_path, bucket_name, minio_path):
        assert os.path.isdir(local_path)

        for local_file in glob.glob(local_path + "/**"):
            local_file = local_file.replace(os.sep, "/")  # Replace \ with / on Windows
            if not os.path.isfile(local_file):
                upload_local_directory_to_minio(
                    local_file,
                    bucket_name,
                    minio_path + "/" + os.path.basename(local_file),
                )
            else:
                content_type = "application/octet-stream"
                if local_file.endswith("png"):
                    content_type = "image/png"
                if local_file.endswith("mp4"):
                    content_type = "video/mp4"
                if local_file.endswith("jpg"):
                    content_type = "image/jpg"
                remote_path = os.path.join(
                    minio_path, local_file[1 + len(local_path) :]
                )
                remote_path = remote_path.replace(
                    os.sep, "/"
                )  # Replace \ with / on Windows
                client.fput_object(
                    bucket_name, remote_path, local_file, content_type=content_type
                )

    upload_local_directory_to_minio("/result", "nightmarebot-output", id)


def upload_swinir(id: str):
    from minio import Minio
    import os

    client = Minio(
        "dumb.dev",
        access_key=os.getenv("NIGHTMAREBOT_MINIO_KEY"),
        secret_key=os.getenv("NIGHTMAREBOT_MINIO_SECRET"),
    )
    client.fput_object(
        "nightmarebot-output",
        f"{id}/output.png",
        "/result/input_SwinIR.png",
        content_type="image/png",
    )


def pixray_prepare(id: str):
    from minio import Minio
    import yaml
    import sys
    import os
    import requests

    client = Minio(
        "dumb.dev",
        access_key=os.getenv("NIGHTMAREBOT_MINIO_KEY"),
        secret_key=os.getenv("NIGHTMAREBOT_MINIO_SECRET"),
    )
    os.makedirs("/tmp/pixray")
    outfile = os.path.join("/tmp/pixray", "input.yaml")
    configfile = os.path.join("/tmp/pixray", "config.yaml")
    contextfile = os.path.join("/tmp/pixray", "context.json")
    client.fget_object("nightmarebot-workflow", f"{id}/input.yaml", outfile)
    client.fget_object("nightmarebot-workflow", f"{id}/context.json", contextfile)
    client.fget_object(
        "nightmarebot-workflow", f"{id}/prompt.txt", "/tmp/pixray/prompt.txt"
    )
    client.fget_object("nightmarebot-workflow", f"{id}/id.txt", "/tmp/pixray/id.txt")
    with open(outfile) as stream:
        try:
            request_settings = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print("Problem with settings", exc)
            sys.exit(1)
    try:
        if (
            request_settings["init_image"]
            and not str(request_settings["init_image"]).isspace()
        ):
            outfile = os.path.join("/tmp/pixray", "init_image.png")
            with requests.get(request_settings["init_image"], stream=True) as r:
                with open(outfile, "wb") as f:
                    for chunk in r.iter_content(chunk_size=16 * 1024):
                        f.write(chunk)
            request_settings["init_image"] = outfile
    except Exception as e:
        print(f"Error saving init_image:{e}", flush=True)

    with open(configfile, "w") as outstream:
        yaml.safe_dump(request_settings, outstream)


def enhance_prepare(id: str):
    from minio import Minio
    import os, sys

    client = Minio(
        "dumb.dev",
        access_key=os.getenv("NIGHTMAREBOT_MINIO_KEY"),
        secret_key=os.getenv("NIGHTMAREBOT_MINIO_SECRET"),
    )
    os.makedirs("/tmp/enhance/lq")
    client.fget_object(
        "nightmarebot-workflow", f"{id}/input.png", "/tmp/enhance/lq/input.png"
    )
    client.fget_object(
        "nightmarebot-workflow", f"{id}/context.json", "/tmp/enhance/context.json"
    )
    client.fget_object(
        "nightmarebot-workflow", f"{id}/prompt.txt", "/tmp/enhance/prompt.txt"
    )
    client.fget_object("nightmarebot-workflow", f"{id}/id.txt", "/tmp/enhance/id.txt")


def majesty_prepare(id: str):
    from minio import Minio
    import os, sys

    client = Minio(
        "dumb.dev",
        access_key=os.getenv("NIGHTMAREBOT_MINIO_KEY"),
        secret_key=os.getenv("NIGHTMAREBOT_MINIO_SECRET"),
    )
    os.makedirs("/tmp/majesty")
    client.fget_object(
        "nightmarebot-workflow", f"{id}/input.json", "/tmp/majesty/input.json"
    )
    client.fget_object(
        "nightmarebot-workflow", f"{id}/context.json", "/tmp/majesty/context.json"
    )
    client.fget_object(
        "nightmarebot-workflow", f"{id}/prompt.txt", "/tmp/majesty/prompt.txt"
    )
    client.fget_object("nightmarebot-workflow", f"{id}/id.txt", "/tmp/majesty/id.txt")
    client.fget_object(
        "nightmarebot-workflow", f"{id}/settings.cfg", "/tmp/majesty/settings.cfg"
    )


def prepare(id: str):
    from minio import Minio
    import os, sys

    client = Minio(
        "dumb.dev",
        access_key=os.getenv("NIGHTMAREBOT_MINIO_KEY"),
        secret_key=os.getenv("NIGHTMAREBOT_MINIO_SECRET"),
    )
    os.makedirs("/input")
    client.fget_object("nightmarebot-workflow", f"{id}/input.json", "/input/input.json")
    client.fget_object(
        "nightmarebot-workflow", f"{id}/context.json", "/input/context.json"
    )
    client.fget_object("nightmarebot-workflow", f"{id}/prompt.txt", "/input/prompt.txt")
    client.fget_object("nightmarebot-workflow", f"{id}/id.txt", "/input/id.txt")


app = App()


@app.subscribe(pubsub_name="jetstream-pubsub", topic="request.majesty-diffusion")
def ldm(event: v1.Event) -> None:
    import os
    import json

    data = json.loads(event.Data())
    id = data["id"]

    try:
        ws = WorkflowService(
            host="https://argo-server.argo:2746",
            token="",
            verify_ssl=False,
            namespace="argo",
        )
        w = Workflow(f"majesty-{id}", ws, parallelism=1, namespace="argo")
        gke_k80_gpu = {"cloud.google.com/gke-accelerator": "nvidia-tesla-k80"}
        gke_t4_gpu = {"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"}
        #    claim_spec = PersistentVolumeClaimSpec(access_modes="ReadWriteMany",claim_name="majesty-models")
        #    claim_meta = ObjectMeta(name="majesty-model")
        #    setattr(w.spec, 'volume_claim_templates', [PersistentVolumeClaim(spec=claim_spec,metadata=claim_meta)])
        command = [
            "python3",
            "latent.py",
            "-c",
            "/tmp/majesty/settings.cfg",
            "-o",
            "/tmp/results",
        ]

        minio_key = str(os.getenv("NIGHTMAREBOT_MINIO_KEY"))
        minio_secret = str(os.getenv("NIGHTMAREBOT_MINIO_SECRET"))
        bot_token = str(os.getenv("NIGHTMAREBOT_TOKEN"))

        p_t = Task(
            "majesty-prepare",
            majesty_prepare,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest",
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
            ],
            output_artifacts=[OutputArtifact(name="input", path="/tmp/majesty")],
            retry=Retry(total=5),
        )

        d_t = Task(
            "majesty-dreamer",
            image=majesty_image,
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=command,
            resources=Resources(
                gpus=1,
                min_mem="16Gi",
                min_cpu="2",
                volumes=[
                    ExistingVolume(name="majesty-models", mount_path="/src/models"),
                    ExistingVolume(name="majesty-cache", mount_path="/root/.cache"),
                ],
            ),
            tolerations=[GPUToleration],
            node_selectors=gke_t4_gpu,
            input_artifacts=[
                InputArtifact(
                    name="input",
                    path="/tmp/majesty",
                    from_task="majesty-prepare",
                    artifact_name="input",
                )
            ],
            output_artifacts=[OutputArtifact(name="result", path="/tmp/results/")],
            retry=Retry(total=5),
        )

        u_t = Task(
            "majesty-upload",
            result_upload,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest",
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
            ],
            input_artifacts=[
                InputArtifact(
                    name="result",
                    path="/result",
                    from_task="majesty-dreamer",
                    artifact_name="result",
                )
            ],
            retry=Retry(total=5),
        )

        r_t = Task(
            "majesty-respond",
            image=publish_image,
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=["dotnet", "NightmareBot.Publish.dll"],
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
            ],
            input_artifacts=[
                InputArtifact(
                    name="input",
                    path="/tmp/majesty",
                    from_task="majesty-prepare",
                    artifact_name="input",
                ),
                InputArtifact(
                    name="result",
                    path="/result/majesty",
                    from_task="majesty-dreamer",
                    artifact_name="result",
                ),
            ],
            retry=Retry(total=5),
        )

        p_t >> d_t >> u_t >> r_t
        w.add_tasks(p_t, d_t, u_t, r_t)
        w.create()
    except Exception as e:
        print(f"Error enqueing request:{e}", flush=True)


@app.subscribe(pubsub_name="jetstream-pubsub", topic="request.latent-diffusion")
def latentDiffusion(event: v1.Event) -> None:
    import os
    import json

    data = json.loads(event.Data())
    id = data["id"]

    try:
        ws = WorkflowService(
            host="https://argo-server.argo:2746",
            token="",
            verify_ssl=False,
            namespace="argo",
        )
        w = Workflow(f"latent-diffusion-{id}", ws, parallelism=1, namespace="argo")
        gke_k80_gpu = {"cloud.google.com/gke-accelerator": "nvidia-tesla-k80"}
        gke_t4_gpu = {"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"}

        command = [
            "python",
            "scripts/txt2img.py",
            "--prompt",
            data["input"]["prompt"],
            "--ddim_steps",
            data["input"]["ddim_steps"],
            "--ddim_eta",
            data["input"]["ddim_eta"],
            "--n_iter",
            data["input"]["n_iter"],
            "--n_samples",
            data["input"]["n_samples"],
            "--scale",
            data["input"]["scale"],
            "--H",
            data["input"]["height"],
            "--W",
            data["input"]["width"],
            "--outdir",
            "/result",
            "--grid_filename",
            f"{id}.png",
        ]
        if data["input"]["plms"]:
            command.append("--plms")

        minio_key = str(os.getenv("NIGHTMAREBOT_MINIO_KEY"))
        minio_secret = str(os.getenv("NIGHTMAREBOT_MINIO_SECRET"))
        bot_token = str(os.getenv("NIGHTMAREBOT_TOKEN"))

        p_t = Task(
            "prepare",
            prepare,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest",
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
            ],
            output_artifacts=[OutputArtifact(name="input", path="/input")],
        )

        d_t = Task(
            "dream",
            image=latent_diffusion_image,
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=command,
            resources=Resources(gpus=1, min_mem="16Gi", min_cpu="2"),
            tolerations=[GPUToleration],
            # node_selectors=gke_t4_gpu,
            input_artifacts=[
                InputArtifact(
                    name="input",
                    path="/result",
                    from_task="prepare",
                    artifact_name="input",
                )
            ],
            output_artifacts=[OutputArtifact(name="result", path="/result")],
        )

        u_t = Task(
            "upload",
            result_upload,
            [{"id": id}],
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest",
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
            ],
            input_artifacts=[
                InputArtifact(
                    name="result",
                    path="/result",
                    from_task="dream",
                    artifact_name="result",
                )
            ],
        )

        r_t = Task(
            "respond",
            image=publish_image,
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=["dotnet", "NightmareBot.Publish.dll"],
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
            ],
            input_artifacts=[
                InputArtifact(
                    name="input",
                    path="/input",
                    from_task="prepare",
                    artifact_name="input",
                ),
                InputArtifact(
                    name="result",
                    path="/result/latent-diffusion",
                    from_task="dream",
                    artifact_name="result",
                ),
            ],
        )

        p_t >> d_t >> u_t >> r_t
        w.add_tasks(p_t, d_t, u_t, r_t)
        w.create()

    except Exception as e:
        print(f"Error enqueing request:{e}", flush=True)


@app.subscribe(pubsub_name="jetstream-pubsub", topic="request.swinir")
def enhance(event: v1.Event) -> None:
    import os
    import json

    data = json.loads(event.Data())
    id = data["id"]

    try:
        ws = WorkflowService(
            host="https://argo-server.argo:2746",
            token="",
            verify_ssl=False,
            namespace="argo",
        )
        w = Workflow(f"enhance-{id}", ws, parallelism=1, namespace="argo")
        gke_k80_gpu = {"cloud.google.com/gke-accelerator": "nvidia-tesla-k80"}
        gke_t4_gpu = {"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"}

        command = [
            "python3",
            "main_test_swinir.py",
            "--task",
            "real_sr",
            "--scale",
            "4",
            "--large_model",
            "--model_path",
            "model_zoo/swinir/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
            "--folder_lq",
            "/tmp/enhance/lq",
        ]

        minio_key = str(os.getenv("NIGHTMAREBOT_MINIO_KEY"))
        minio_secret = str(os.getenv("NIGHTMAREBOT_MINIO_SECRET"))
        bot_token = str(os.getenv("NIGHTMAREBOT_TOKEN"))

        p_t = Task(
            "swinir-prepare",
            enhance_prepare,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest",
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
            ],
            output_artifacts=[OutputArtifact(name="input", path="/tmp/enhance")],
        )

        e_t = Task(
            "swinir-enhance",
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/swinir-enhance:latest",
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=command,
            resources=Resources(gpus=1, min_mem="8Gi", min_cpu="2"),
            tolerations=[GPUToleration],
            # node_selectors=gke_k80_gpu,
            input_artifacts=[
                InputArtifact(
                    name="input",
                    path="/tmp/enhance",
                    from_task="swinir-prepare",
                    artifact_name="input",
                )
            ],
            output_artifacts=[
                OutputArtifact(
                    name="result", path="/src/results/swinir_real_sr_x4_large/"
                )
            ],
        )

        u_t = Task(
            "swinir-upload",
            upload_swinir,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest",
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
            ],
            input_artifacts=[
                InputArtifact(
                    name="result",
                    path="/result",
                    from_task="swinir-enhance",
                    artifact_name="result",
                )
            ],
        )

        r_t = Task(
            "swinir-respond",
            image=publish_image,
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=["dotnet", "NightmareBot.Publish.dll"],
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
            ],
            input_artifacts=[
                InputArtifact(
                    name="input",
                    path="/result/swinir",
                    from_task="swinir-prepare",
                    artifact_name="input",
                ),
                InputArtifact(
                    name="result",
                    path="/result/swinir/output",
                    from_task="swinir-enhance",
                    artifact_name="result",
                ),
            ],
        )

        p_t >> e_t >> u_t >> r_t
        w.add_tasks(p_t, e_t, u_t, r_t)
        w.create()
    except Exception as e:
        print(f"Error enqueing request:{e}", flush=True)


@app.subscribe(pubsub_name="jetstream-pubsub", topic="request.esrgan")
def esrgan(event: v1.Event) -> None:
    import os
    import json

    data = json.loads(event.Data())
    id = data["id"]

    try:
        ws = WorkflowService(
            host="https://argo-server.argo:2746",
            token="",
            verify_ssl=False,
            namespace="argo",
        )
        w = Workflow(f"enhance-{id}", ws, parallelism=1, namespace="argo")
        gke_k80_gpu = {"cloud.google.com/gke-accelerator": "nvidia-tesla-k80"}
        gke_t4_gpu = {"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"}

        command = [
            "python",
            "inference_realesrgan.py",
            "-n",
            "RealESRGAN_x4plus",
            "-i",
            "/input/lq",
            "-o",
            "/result",
        ]

        if data["input"]["face_enhance"]:
            command.append("--face_enhance")

        if data["input"]["outscale"]:
            command.append("--outscale")
            command.append(data["input"]["outscale"])

        minio_key = str(os.getenv("NIGHTMAREBOT_MINIO_KEY"))
        minio_secret = str(os.getenv("NIGHTMAREBOT_MINIO_SECRET"))
        bot_token = str(os.getenv("NIGHTMAREBOT_TOKEN"))

        p_t = Task(
            "prepare",
            enhance_prepare,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest",
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
            ],
            output_artifacts=[OutputArtifact(name="input", path="/tmp/enhance")],
        )

        e_t = Task(
            "enhance",
            image=esrgan_image,
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=command,
            resources=Resources(gpus=1, min_mem="16Gi", min_cpu="2"),
            tolerations=[GPUToleration],
            # node_selectors=gke_k80_gpu,
            input_artifacts=[
                InputArtifact(
                    name="input",
                    path="/input",
                    from_task="prepare",
                    artifact_name="input",
                )
            ],
            output_artifacts=[OutputArtifact(name="result", path="/result")],
        )

        u_t = Task(
            "upload",
            result_upload,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest",
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
            ],
            input_artifacts=[
                InputArtifact(
                    name="result",
                    path="/result",
                    from_task="enhance",
                    artifact_name="result",
                )
            ],
        )

        r_t = Task(
            "respond",
            image=publish_image,
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=["dotnet", "NightmareBot.Publish.dll"],
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
            ],
            input_artifacts=[
                InputArtifact(
                    name="input",
                    path="/input",
                    from_task="prepare",
                    artifact_name="input",
                ),
                InputArtifact(
                    name="result",
                    path="/result/enhance",
                    from_task="enhance",
                    artifact_name="result",
                ),
            ],
        )

        p_t >> e_t >> u_t >> r_t
        w.add_tasks(p_t, e_t, u_t, r_t)
        w.create()

    except Exception as e:
        print(f"Error enqueing request:{e}", flush=True)


@app.subscribe(pubsub_name="jetstream-pubsub", topic="request.pixray")
def dream(event: v1.Event) -> None:
    import os
    import json

    data = json.loads(event.Data())
    id = data["id"]

    try:
        ws = WorkflowService(
            host="https://argo-server.argo:2746",
            token="",
            verify_ssl=False,
            namespace="argo",
        )
        w = Workflow(f"pixray-{id}", ws, parallelism=1, namespace="argo")
        gke_t4_gpu = {"cloud.google.com/gke-accelerator": "nvidia-tesla-t4"}

        command = ["python", "draw.py", "/tmp/pixray"]

        minio_key = str(os.getenv("NIGHTMAREBOT_MINIO_KEY"))
        minio_secret = str(os.getenv("NIGHTMAREBOT_MINIO_SECRET"))
        bot_token = str(os.getenv("NIGHTMAREBOT_TOKEN"))

        p_t = Task(
            "pixray-prepare",
            pixray_prepare,
            [{"id": id}],
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest",
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
            ],
            output_artifacts=[OutputArtifact(name="input", path="/tmp/pixray")],
        )

        d_t = Task(
            "pixray-dreamer",
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/pixray-dreamer:latest",
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=command,
            resources=Resources(gpus=1, min_mem="16Gi", min_cpu="2"),
            tolerations=[GPUToleration],
            node_selectors=gke_t4_gpu,
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
            ],
            input_artifacts=[
                InputArtifact(
                    name="input",
                    path="/tmp/pixray",
                    from_task="pixray-prepare",
                    artifact_name="input",
                )
            ],
            output_artifacts=[OutputArtifact(name="result", path="/tmp/pixray")],
        )

        u_t = Task(
            "pixray-upload",
            result_upload,
            [{"id": id}],
            image="us-docker.pkg.dev/nightmarebot-ai/nightmarebot/hera-dreamer:latest",
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
            ],
            input_artifacts=[
                InputArtifact(
                    name="result",
                    path="/result",
                    from_task="pixray-dreamer",
                    artifact_name="result",
                )
            ],
        )

        r_t = Task(
            "pixray-respond",
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            image=publish_image,
            command=["dotnet", "NightmareBot.Publish.dll"],
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
            ],
            input_artifacts=[
                InputArtifact(
                    name="result",
                    path="/result/pixray",
                    from_task="pixray-dreamer",
                    artifact_name="result",
                )
            ],
        )

        p_t >> d_t >> u_t >> r_t
        w.add_tasks(p_t, d_t, u_t, r_t)
        print(f'creating workflow for prompt {data["input"]["prompts"]}', flush=True)
        w.create()
    except Exception as e:
        print(f"Error enqueing request:{e}", flush=True)


app.run(50055)
