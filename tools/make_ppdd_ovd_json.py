"""
Copyright (c) 2022
All rights reserved.
For full license text see https://ovad-benchmark.github.io/
By Maria A. Bravo

This file contains functions to parse json annotation files and builds the training json
"""
import os
import argparse
import json
from collections import defaultdict
import sys

sys.path.insert(0, os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path", default="datasets/coco/annotations/instances_val2017.json"
    )
    parser.add_argument(
        "--cat_path", default="datasets/coco/annotations/instances_val2017.json"
    )
    parser.add_argument("--save_json_path", default="")
    parser.add_argument("--base_novel", default="all")
    parser.add_argument("--convert_caption", action="store_true")
    args = parser.parse_args()

    # Load all categories from file
    print("Loading", args.cat_path)
    cat = json.load(open(args.cat_path, "r"))["categories"]

    # load annotation file
    print("Loading", args.json_path)
    data = json.load(open(args.json_path, "r"))

    # if caption file
    if args.convert_caption:
        num_caps = 0
        caps = defaultdict(list)
        for x in data["annotations"]:
            caps[x["image_id"]].append(x["caption"])
        for x in data["images"]:
            x["captions"] = caps[x["id"]]
            num_caps += len(x["captions"])
        print("# captions", num_caps)
        data["annotations"] = []
        save_json_path = os.path.join(
            os.path.dirname(args.json_path),
            "{set_name}_categories.json".format(
                set_name=os.path.basename(args.json_path).replace(".json", ""),
            ),
        )

    # if instance file
    else:
        if args.base_novel != "all":
            if "/ppdd/" in args.json_path:
                from ovadb.data.datasets.utils_ppdd import (
                    ppdd_categories_base,
                    ppdd_categories_novel
                )

                valid_ids = []
                if "base" in args.base_novel:
                    valid_ids.extend([x["id"] for x in ppdd_categories_base])
                if "novel" in args.base_novel:
                    valid_ids.extend([x["id"] for x in ppdd_categories_novel])
                
                # oldid_to_newid = {c:i for i, c in enumerate(valid_ids)}

                # filter annotation file
                filtered_images = []
                filtered_annotations = []
                useful_image_ids = set()

                for ann in data["annotations"]:
                    if ann["category_id"] in valid_ids:
                        # ann["category_id"] = oldid_to_newid[ann["category_id"]]
                        filtered_annotations.append(ann)
                        useful_image_ids.add(ann["image_id"])

                for img in data["images"]:
                    if img["id"] in useful_image_ids:
                        filtered_images.append(img)

            data["annotations"] = filtered_annotations
            data["images"] = filtered_images
            _new_cat = []
            for c in cat:
                if c["id"] in valid_ids:
                    _new_cat.append(c)
            data["categories"] = _new_cat

        else:
            data["categories"] = cat



        # insert attributes
        from ovadb.data.datasets.utils_ppdd import (
                    classid_to_attr
                )

        # 중복 없는 속성 리스트 생성
        unique_attrs = set()
        for attrs in classid_to_attr.values():
            unique_attrs.update(attrs)

        # 속성 리스트를 정렬하고 ID 할당
        unique_attrs = sorted(unique_attrs)
        attribute_json_list = []

        for idx, full_attr in enumerate(unique_attrs):
            attr_type, attr_name = full_attr.split(":", 1)
            attribute_json_list.append({
                "id": idx,
                "name": full_attr,
                "type": attr_type,
                "parent_type": attr_type,
                "is_has_att": "has"  # default 지정 (변경 가능)
            })
        data["attributes"] = attribute_json_list
        
        save_json_path = os.path.join(
            os.path.dirname(args.json_path),
            "{set_name}_{base_novel}.json".format(
                set_name=os.path.basename(args.json_path).replace(".json", ""),
                base_novel=args.base_novel,
            ),
        )

    print("Total images", len(data["images"]))
    print("Total annotations", len(data["annotations"]))
    print("Total categories", len(data["categories"]))

    # save modified json data
    if args.save_json_path != "":
        save_json_path = args.save_json_path
    print("Saving to", save_json_path)

    os.makedirs(os.path.dirname(save_json_path), exist_ok=True)
    json.dump(data, open(save_json_path, "w"))
