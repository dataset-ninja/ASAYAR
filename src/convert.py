import supervisely as sly
import os
from dataset_tools.convert import unpack_if_archive
import src.settings as s
from urllib.parse import unquote, urlparse
from supervisely.io.fs import get_file_name, get_file_size
import shutil
from glob import glob
from tqdm import tqdm
import imagesize
import xml.etree.ElementTree as ET


def download_dataset(teamfiles_dir: str) -> str:
    """Use it for large datasets to convert them on the instance"""
    api = sly.Api.from_env()
    team_id = sly.env.team_id()
    storage_dir = sly.app.get_data_dir()

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, str):
        parsed_url = urlparse(s.DOWNLOAD_ORIGINAL_URL)
        file_name_with_ext = os.path.basename(parsed_url.path)
        file_name_with_ext = unquote(file_name_with_ext)

        sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
        local_path = os.path.join(storage_dir, file_name_with_ext)
        teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

        fsize = api.file.get_directory_size(team_id, teamfiles_dir)
        with tqdm(
            desc=f"Downloading '{file_name_with_ext}' to buffer...",
            total=fsize,
            unit="B",
            unit_scale=True,
        ) as pbar:
            api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)
        dataset_path = unpack_if_archive(local_path)

    if isinstance(s.DOWNLOAD_ORIGINAL_URL, dict):
        for file_name_with_ext, url in s.DOWNLOAD_ORIGINAL_URL.items():
            local_path = os.path.join(storage_dir, file_name_with_ext)
            teamfiles_path = os.path.join(teamfiles_dir, file_name_with_ext)

            if not os.path.exists(get_file_name(local_path)):
                fsize = api.file.get_directory_size(team_id, teamfiles_dir)
                with tqdm(
                    desc=f"Downloading '{file_name_with_ext}' to buffer...",
                    total=fsize,
                    unit="B",
                    unit_scale=True,
                ) as pbar:
                    api.file.download(team_id, teamfiles_path, local_path, progress_cb=pbar)

                sly.logger.info(f"Start unpacking archive '{file_name_with_ext}'...")
                unpack_if_archive(local_path)
            else:
                sly.logger.info(
                    f"Archive '{file_name_with_ext}' was already unpacked to '{os.path.join(storage_dir, get_file_name(file_name_with_ext))}'. Skipping..."
                )

        dataset_path = storage_dir
    return dataset_path


def count_files(path, extension):
    count = 0
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                count += 1
    return count


def xml_to_dict(xml_file_path):
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        return element_to_dict(root)
    except Exception as e:
        print(f"Error parsing XML: {e}")
        return None


def element_to_dict(element):
    result = {}

    if element.text:
        result[element.tag] = element.text

    for child in element:
        child_data = element_to_dict(child)
        if child.tag in result:
            if type(result[child.tag]) is list:
                result[child.tag].append(child_data)
            else:
                result[child.tag] = [result[child.tag], child_data]
        else:
            result[child.tag] = child_data

    return result


def create_ann(image_path):
    labels = []
    img_width, img_height = imagesize.get(image_path)
    subdataset_path = os.path.dirname(os.path.dirname(os.path.dirname(image_path)))
    image_path_splits = image_path.split("/")

    ann_path = os.path.join(
        subdataset_path,
        image_path_splits[-3],
        "Annotations",
        (os.path.basename(image_path)[:-3] + "xml"),
    )
    # different folder structure, have to bidlocode

    if image_path_splits[-4] == "ASAYAR_TXT":
        ann_path = os.path.join(
            subdataset_path,
            "Annotations",
            "Word-Level",
            image_path_splits[-2],
            (os.path.basename(image_path)[:-3] + "xml"),
        )
    ann_dict = xml_to_dict(ann_path)
    if ann_dict is None:
        print(f"Error:{image_path}")
        return sly.Annotation(img_size=(img_height, img_width), labels=labels)
    if type(ann_dict["object"]) == dict:
        objct = []
        objct.append(ann_dict["object"])
    else:
        objct = ann_dict["object"]
    for obj in objct:
        x_min = obj["bndbox"]["xmin"]["xmin"]
        y_min = obj["bndbox"]["ymin"]["ymin"]
        x_max = obj["bndbox"]["xmax"]["xmax"]
        y_max = obj["bndbox"]["ymax"]["ymax"]

        obj_class_name = obj["name"]["name"]
        rectangle = sly.Rectangle(
            top=int(y_min), left=int(x_min), bottom=int(y_max), right=int(x_max)
        )
        obj_class = obj_class_dict.get(obj_class_name)
        if obj_class is None:
            raise ValueError()

        label = sly.Label(rectangle, obj_class)
        labels.append(label)

    tag_value = os.path.basename(subdataset_path).split("_")[1].lower()
    subds_tag = sly.Tag(tm_subds, value=tag_value)

    return sly.Annotation(img_size=(img_height, img_width), labels=labels, img_tags=[subds_tag])


obj_class_dict = {
    "Guide_Sign": sly.ObjClass("guide sign", geometry_type=sly.Rectangle, color=[0, 255, 0]),
    "Regulatory_Sign": sly.ObjClass("regulatory sign", geometry_type=sly.Rectangle),
    "Warning_Sign": sly.ObjClass("warning sign", geometry_type=sly.Rectangle),
    "Down_Arrow": sly.ObjClass("down arrow", geometry_type=sly.Rectangle, color=[255, 255, 0]),
    "Down_Left_Arrow": sly.ObjClass("down left arrow", geometry_type=sly.Rectangle),
    "Down_Right_Arrow": sly.ObjClass(
        "down right arrow", geometry_type=sly.Rectangle, color=[0, 255, 255]
    ),
    "Mixed_Arrow": sly.ObjClass("mixed arrow", geometry_type=sly.Rectangle, color=[255, 0, 0]),
    "Right_Arrow": sly.ObjClass("right arrow", geometry_type=sly.Rectangle, color=[0, 0, 255]),
    "Up_Arrow": sly.ObjClass("up arrow", geometry_type=sly.Rectangle, color=[255, 0, 255]),
    "Arabic_Word": sly.ObjClass("arabic box", geometry_type=sly.Rectangle),
    "Latin_Word": sly.ObjClass("latin box", geometry_type=sly.Rectangle),
    "Number": sly.ObjClass("number", geometry_type=sly.Rectangle),
    "Mixed_Word": sly.ObjClass("mixed box", geometry_type=sly.Rectangle),
}
tm_subds = sly.TagMeta("subdataset", sly.TagValueType.ANY_STRING)
meta = sly.ProjectMeta(obj_classes=list(obj_class_dict.values()), tag_metas=[tm_subds])


def convert_and_upload_supervisely_project(
    api: sly.Api, workspace_id: int, project_name: str
) -> sly.ProjectInfo:
    project = api.project.create(workspace_id, project_name)
    api.project.update_meta(project.id, meta.to_json())

    train_dirs = [
        "/mnt/c/users/german/documents/ASAYAR_SIGN/Train/Images/*",
        "/mnt/c/users/german/documents/ASAYAR_SYM/Train/Images/*",
        "/mnt/c/users/german/documents/ASAYAR_TXT/Images/Train/*",
    ]
    test_dirs = [
        "/mnt/c/users/german/documents/ASAYAR_SIGN/Test/Images/*",
        "/mnt/c/users/german/documents/ASAYAR_SYM/Test/Images/*",
        "/mnt/c/users/german/documents/ASAYAR_TXT/Images/Test/*",
    ]

    test_images = []
    for testdir in test_dirs:
        test_img_pathes = glob(testdir)
        for test_img_path in test_img_pathes:
            test_images.append(test_img_path)

    train_images = []
    for dir in train_dirs:
        train_img_pathes = glob(dir)
        for train_img_path in train_img_pathes:
            train_images.append(train_img_path)

    ds_paths = {
        "train": train_images,
        "test": test_images,
    }

    batch_size = 50
    for ds_name, images_pathes in ds_paths.items():
        dataset = api.dataset.create(project.id, ds_name, change_name_if_conflict=True)
        progress = sly.Progress("Create dataset {}".format(ds_name), len(images_pathes))
        for img_pathes_batch in sly.batched(images_pathes, batch_size=batch_size):
            # ds has the same filenames across every subdataset, so have to change each name
            img_names_batch = [
                (os.path.dirname(im_path)).split("/")[-3].split("_")[1].lower()
                + "_"
                + sly.fs.get_file_name_with_ext(im_path)
                for im_path in img_pathes_batch
            ]

            img_infos = api.image.upload_paths(
                dataset.id,
                img_names_batch,
                img_pathes_batch,
            )
            img_ids = [im_info.id for im_info in img_infos]

            anns = [create_ann(image_path) for image_path in img_pathes_batch]
            api.annotation.upload_anns(img_ids, anns)
    return project
