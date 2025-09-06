import io
from fvcore.common.file_io import PathManager
import contextlib
from detectron2.data.datasets.builtin_meta import _get_coco_instances_meta
from datasets.ovad.ovad import OVAD

import os
from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES, _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog
from ovadb.data.datasets.register_data import register_custom_instances, load_coco_json, load_ppdd_json

from ovadb.data.datasets.utils_ppdd import (
    ppdd_categories_base,
    ppdd_categories_novel,
)


def _get_metadata(cat):

    if cat == "all":
        all_categories = ppdd_categories_base + ppdd_categories_novel
        id_to_name = {x["id"]: x["name"] for x in all_categories}
        id_to_color = {x["id"]: x["color"] for x in all_categories}

        thing_dataset_id_to_contiguous_id = {
            x: i for i, x in enumerate(sorted(id_to_name))
        }

        thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
        thing_colors = [id_to_color[k] for k in sorted(id_to_color)]
        return {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes,
            "thing_colors": thing_colors,
        }
    else:
        id_to_name = {}
        if "base" in cat:
            id_to_name.update({x["id"]: x["name"] for x in ppdd_categories_base})
        if "novel" in cat:
            id_to_name.update({x["id"]: x["name"] for x in ppdd_categories_novel})

        assert len(id_to_name) > 0

        thing_dataset_id_to_contiguous_id = {
            x: i for i, x in enumerate(sorted(id_to_name))
        } 
        thing_classes = [id_to_name[k] for k in sorted(id_to_name)]
        return {
            "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
            "thing_classes": thing_classes,
        }


_PREDEFINED_SPLITS_COCO = {
    
    "ppdd_ovd_train_base": 
        ("ppdd/train", "ppdd/ppdd_ovad/ppdd_train_base.json", "base"),
    "ppdd_ovd_test_base":
        ("ppdd/test", "ppdd/ppdd_ovad/ppdd_test_base.json", "base"),
    "ppdd_ovd_test_novel":
        ("ppdd/test", "ppdd/ppdd_ovad/ppdd_test_novel.json", "novel"),
    "ppdd_ovd_test_all":
        ("ppdd/test", "ppdd/ppdd_ovad/ppdd_test_all.json", "all"),
    "ppdd_ovd_test_all_debug":
        ("ppdd/test", "ppdd/ppdd_ovad/ppdd_test_all_debug.json", "all"),

    # "ppdd_ovad_test_all":
    #     ("ppdd/test", "ppdd/ppdd_ovad/ppdd_test_all.json", "all"),
}


def register_custom_coco_instances(name, metadata, json_file, image_root):
    """
    Register a dataset in COCO's json annotation format for
    instance detection, instance segmentation and keypoint detection.
    (i.e., Type 1 and 2 in http://cocodataset.org/#format-data.
    `instances*.json` and `person_keypoints*.json` in the dataset).

    This is an example of how to register a new dataset.
    You can do something similar to this function, to register new datasets.

    Args:
        name (str): the name that identifies a dataset, e.g. "coco_2014_train".
        metadata (dict): extra metadata associated with this dataset.  You can
            leave it as an empty dict.
        json_file (str): path to the json instance annotation file.
        image_root (str or path-like): directory which contains all the images.
    """
    assert isinstance(name, str), name
    assert isinstance(json_file, (str, os.PathLike)), json_file
    assert isinstance(image_root, (str, os.PathLike)), image_root
    if name not in DatasetCatalog and name not in MetadataCatalog:

        # 1. register a function which returns dicts
        DatasetCatalog.register(
            name, lambda: load_ppdd_json(json_file, image_root, name, ["att_vec"])
        )

        # 2. Optionally, add metadata about this dataset,
        # since they might be useful in evaluation, visualization or logging
        MetadataCatalog.get(name).set(
            json_file=json_file,
            image_root=image_root,
            evaluator_type="coco",
            **metadata
        )


for key, (image_root, json_file, cat) in _PREDEFINED_SPLITS_COCO.items():
    register_custom_coco_instances(
        key,
        _get_metadata(cat),
        os.path.join("datasets", json_file) if "://" not in json_file else json_file,
        os.path.join("datasets", image_root),
    )


def _get_ovad_meta(dataset_name, ann_file):
    metadata = _get_coco_instances_meta()
    ann_file = PathManager.get_local_path(ann_file)
    with contextlib.redirect_stdout(io.StringIO()):
        ovad_api = OVAD(ann_file)

    # Make the dictionaries for evaluator of attributes
    att2idx = {}
    idx2att = {}
    attr_type = {}
    attr_parent_type = {}
    # attribute_head_tail = {"head": set(), "medium": set(), "tail": set()}

    for att in ovad_api.atts.values():
        att2idx[att["name"]] = att["id"]
        idx2att[att["id"]] = att["name"]

        if att["type"] not in attr_type.keys():
            attr_type[att["type"]] = set()
        attr_type[att["type"]].add(att["name"])

        if att["parent_type"] not in attr_parent_type.keys():
            attr_parent_type[att["parent_type"]] = set()
        attr_parent_type[att["parent_type"]].add(att["type"])

        # attribute_head_tail[att["freq_set"]].add(att["name"])

    attr_type = {key: list(val) for key, val in attr_type.items()}
    attr_parent_type = {key: list(val) for key, val in attr_parent_type.items()}
    # attribute_head_tail = {key: list(val) for key, val in attribute_head_tail.items()}

    attribute_list = list(att2idx.keys())
    attCount = {att: 0 for att in attribute_list}

    metadata["attribute_classes"] = attribute_list
    metadata["att2idx"] = att2idx
    metadata["idx2att"] = idx2att
    metadata["att_base_novel"] = {}
    metadata["att_type"] = attr_type
    metadata["att_parent_type"] = attr_parent_type
    evaluator_type = "attribute"
    if "boxann" in dataset_name:
        evaluator_type += "_boxann"
    metadata["evaluator_type"] = evaluator_type
    # metadata["attribute_head_tail"] = attribute_head_tail

    # Add object information in metadata
    cat_ids = list(ovad_api.cats.keys())
    cats = list(ovad_api.cats.values())
    # The categories in a custom json file may not be sorted.
    thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    metadata["thing_classes"] = thing_classes
    # In COCO, certain category ids are artificially removed,
    # and by convention they are always ignored.
    # We deal with COCO's id issue and translate
    # the category ids to contiguous ids in [0, 80).

    # It works by looking at the "categories" field in the json, therefore
    # if users' own json also have incontiguous ids, we'll
    # apply this mapping as well but print a warning.
    id_map = {v: i for i, v in enumerate(cat_ids)}
    metadata["thing_dataset_id_to_contiguous_id"] = id_map

    return metadata

for key, (image_root, json_file, cat) in _PREDEFINED_SPLITS_COCO.items():
    if "ppdd_ovad" in key:
        extra_annotation_keys = ["att_vec", "id"]
        register_custom_instances(
            key,
            _get_ovad_meta(key, os.path.join("datasets", json_file)),
            os.path.join("datasets", json_file) if "://" not in json_file else json_file,
            os.path.join("datasets", image_root),
            extra_annotation_keys,
        )