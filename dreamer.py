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
import os

from cloudevents.sdk.event import v1
from dapr.clients import DaprClient
from dapr.ext.grpc import App


publish_image = "ghcr.io/nightmareai/nightmarebot-publish@sha256:7a365e631866a21b22361661f7f373cafa3b7d78f92b28f88c86c1651250160c"
majesty_image = "ghcr.io/nightmareai/majesty-diffusion@sha256:fb81446fcc13fa1a048c3d683e38113cae114aa5216218a639c03946b1f88dcb"
pixray_image = "ghcr.io/nightmareai/pixray@sha256:4789d76512bbe483ef0c50023366f684e8f8775f59bf133f687761d3913e09e0"
latent_diffusion_image = "ghcr.io/nightmareai/latent-diffusion@sha256:fc529d49066fc2d2764c42a66dd9e68a698cd9f67c76e262740edbfa9f8ca914"
esrgan_image = "ghcr.io/nightmareai/real-esrgan@sha256:306b465203ad35a02526b249b0e42da22f38831a41570661a9d18a0334c8d26f"
swinir_image = "ghcr.io/nightmareai/swinir@sha256:11aec61fa66b568630cdde5bf32539c5bf0e44425a7b135361bd77eeb697742e"
dreamer_image = "ghcr.io/nightmareai/dreamer:main"


minio_key = str(os.getenv("NIGHTMAREBOT_MINIO_KEY"))
minio_secret = str(os.getenv("NIGHTMAREBOT_MINIO_SECRET"))
bot_token = str(os.getenv("NIGHTMAREBOT_TOKEN"))
openai_key = str(os.getenv("OPENAI_SECRET_KEY"))


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
    import os

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

        command = [
            "python3",
            "latent.py",
            "-c",
            "/tmp/majesty/settings.cfg",
            "-o",
            "/tmp/results",
            "--enable_aesthetic_embeddings",
            "--model_source",
            "https://storage.googleapis.com/majesty-diffusion-models",
        ]

        p_t = Task(
            "majesty-prepare",
            majesty_prepare,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            # annotations={"multicluster.admiralty.io/elect": ""},
            image=dreamer_image,
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
            ],
            output_artifacts=[OutputArtifact(name="input", path="/tmp/majesty")],
            retry=Retry(limit=3),
        )

        d_t = Task(
            "majesty-dreamer",
            image=majesty_image,
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=command,
            # annotations={"multicluster.admiralty.io/elect": ""},
            resources=Resources(
                gpus=1,
                min_mem="16Gi",
                min_cpu="2",
                volumes=[
                    ExistingVolume(
                        name="majesty-models-local", mount_path="/src/models"
                    ),
                    ExistingVolume(
                        name="majesty-cache-local", mount_path="/root/.cache"
                    ),
                ],
            ),
            tolerations=[GPUToleration],
            node_selectors={"dreamer.nightmarebot.com/majesty": "true"},
            input_artifacts=[
                InputArtifact(
                    name="input",
                    path="/tmp/majesty",
                    from_task="majesty-prepare",
                    artifact_name="input",
                )
            ],
            output_artifacts=[OutputArtifact(name="result", path="/tmp/results/")],
            retry=Retry(limit=3),
        )

        u_t = Task(
            "majesty-upload",
            result_upload,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            # annotations={"multicluster.admiralty.io/elect": ""},
            image=dreamer_image,
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
            retry=Retry(limit=3),
        )

        r_t = Task(
            "majesty-respond",
            image=publish_image,
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            # annotations={"multicluster.admiralty.io/elect": ""},
            command=["dotnet", "NightmareBot.Publish.dll"],
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
                EnvSpec(name="OPENAI_SECRET_KEY", value=openai_key),
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
            retry=Retry(limit=3),
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

        p_t = Task(
            "prepare",
            prepare,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            # annotations={"multicluster.admiralty.io/elect": ""},
            image=dreamer_image,
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
            # annotations={"multicluster.admiralty.io/elect": ""},
            resources=Resources(gpus=1, min_mem="16Gi", min_cpu="2"),
            tolerations=[GPUToleration],
            node_selectors={"dreamer.nightmarebot.com/latent": "true"},
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
            image=dreamer_image,
            # annotations={"multicluster.admiralty.io/elect": ""},
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
            # annotations={"multicluster.admiralty.io/elect": ""},
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
                EnvSpec(name="OPENAI_SECRET_KEY", value=openai_key),
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

        p_t = Task(
            "swinir-prepare",
            enhance_prepare,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            # annotations={"multicluster.admiralty.io/elect": ""},
            image=dreamer_image,
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
            ],
            output_artifacts=[OutputArtifact(name="input", path="/tmp/enhance")],
        )

        e_t = Task(
            "swinir-enhance",
            image=swinir_image,
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=command,
            # annotations={"multicluster.admiralty.io/elect": ""},
            resources=Resources(gpus=1, min_mem="8Gi", min_cpu="2"),
            tolerations=[GPUToleration],
            node_selectors={"enhance.nightmarebot.com/swinir": "true"},
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
            # annotations={"multicluster.admiralty.io/elect": ""},
            image=dreamer_image,
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
            # annotations={"multicluster.admiralty.io/elect": ""},
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
                EnvSpec(name="OPENAI_SECRET_KEY", value=openai_key),
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

        p_t = Task(
            "prepare",
            enhance_prepare,
            [{"id": id}],
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            # annotations={"multicluster.admiralty.io/elect": ""},
            image=dreamer_image,
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
            # annotations={"multicluster.admiralty.io/elect": ""},
            tolerations=[GPUToleration],
            node_selectors={"enhance.nightmarebot.com/esrgan": "true"},
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
            # annotations={"multicluster.admiralty.io/elect": ""},
            image=dreamer_image,
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
            # annotations={"multicluster.admiralty.io/elect": ""},
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
                EnvSpec(name="OPENAI_SECRET_KEY", value=openai_key),
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

        p_t = Task(
            "pixray-prepare",
            pixray_prepare,
            [{"id": id}],
            image=dreamer_image,
            # annotations={"multicluster.admiralty.io/elect": ""},
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
            ],
            output_artifacts=[OutputArtifact(name="input", path="/tmp/pixray")],
        )

        d_t = Task(
            "pixray-dreamer",
            image=pixray_image,
            image_pull_policy=ImagePullPolicy.IfNotPresent,
            command=command,
            resources=Resources(gpus=1, min_mem="16Gi", min_cpu="2"),
            tolerations=[GPUToleration],
            # annotations={"multicluster.admiralty.io/elect": ""},
            node_selectors={"dreamer.nightmarebot.com/pixray": "true"},
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
            image=dreamer_image,
            # annotations={"multicluster.admiralty.io/elect": ""},
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
            # annotations={"multicluster.admiralty.io/elect": ""},
            env_specs=[
                EnvSpec(name="NIGHTMAREBOT_MINIO_KEY", value=minio_key),
                EnvSpec(name="NIGHTMAREBOT_MINIO_SECRET", value=minio_secret),
                EnvSpec(name="NIGHTMAREBOT_TOKEN", value=bot_token),
                EnvSpec(name="OPENAI_SECRET_KEY", value=openai_key),
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
