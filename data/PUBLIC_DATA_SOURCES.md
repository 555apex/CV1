# 公开数据样例说明

本项目的合成数据用于可控真值实验；公开样例和后续真实照片用于验证输入、角点标注和视觉效果。

## 推荐来源

- SmartDoc 2017 官方测试集：
  https://github.com/smartdoc2017-competition/test_set_with_ground_truth_and_extras
- SmartDoc 官方组织页：
  https://github.com/smartdoc2017-competition
- Doc3D 官方仓库（主要用于弯曲文档展开，不作为平面单应变换主数据集）：
  https://github.com/cvlab-stonybrook/doc3D-dataset

- SmartDoc 2017 sample/demo release:
  https://github.com/smartdoc2017-competition/sample_dataset/releases/tag/1.0

The project includes five small samples from the release: `poster01`, `paper01`, `receipt01`, `card01`, and `screen01`. We keep only reference frames, ground-truth images, and converted corner JSON files; the original 469 MB archive is not committed.

The five selected cases keep all four corners inside the reference frame so they match the current strict input validation. The release also contains cases with a corner outside the frame; those are intentionally deferred until the pipeline gains an explicit partial-document policy.

- `task_data.json` provides the target size, reference frame id, and four corners. The labels are converted to the project order `tl, tr, br, bl`.
- The release README credits CVC-UAB and L3i-ULR. The repository API does not declare a machine-readable license for this release, so the sample JSON keeps the attribution note and users should follow the upstream terms before redistribution.
- Re-download the archive with `code/fetch_smartdoc_sample.py`; the helper supports resumable byte-range downloads.

## 本地放置约定

将少量、允许课程提交的公开样例放入：

- 图像：`data/oblique/<sample_name>.<png|jpg|jpeg>`
- 角点：`data/corners/<sample_name>.json`

角点 JSON 至少包含：

```json
{
  "order": "tl, tr, br, bl",
  "corners": [[x0, y0], [x1, y1], [x2, y2], [x3, y3]],
  "source": "SmartDoc 2017",
  "license": "see upstream dataset terms",
  "source_url": "https://github.com/smartdoc2017-competition/test_set_with_ground_truth_and_extras",
  "image_size": [width, height]
}
```

如果许可证不允许重新分发原图，只提交本文件、样例编号、来源链接和下载说明，不把原始图片加入仓库。
