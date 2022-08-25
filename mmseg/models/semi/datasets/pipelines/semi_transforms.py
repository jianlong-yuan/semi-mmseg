import cv2, random, copy, mmcv, inspect, torch
import numpy as np
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines.compose import Compose
from mmcv.utils import build_from_cfg
from torchvision import transforms as _transforms
import albumentations

@PIPELINES.register_module()
class StrongWeakAug(object):
    # add by yuanjianlong
    def __init__(self, pre_transforms, weak_transforms, strong_transforms):
        self.pre_transforms=Compose(pre_transforms)
        self.weak_transforms = Compose(weak_transforms)
        self.strong_transforms = Compose(strong_transforms)

    def __call__(self, results, **kwargs):
        tmp = self.pre_transforms(results)
        weak = self.weak_transforms(copy.deepcopy(tmp))
        strong = self.strong_transforms(copy.deepcopy(tmp))
        return {"weak": weak, "strong": strong}


@PIPELINES.register_module()
class SetIgnoreSeg(object):
    # add by yuanjianlong
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index

    def __call__(self, results):
        img = results['img']
        results['gt_semantic_seg'] = np.zeros(img.shape[:2]) + self.ignore_index
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'set_ignore_index={self.ignore_index}, '
        return repr_str


@PIPELINES.register_module()
class RandomAppliedTrans(object):
    """Randomly applied transformations.
    Args:
        transforms (list[dict]): List of transformations in dictionaries.
        p (float, optional): Probability. Defaults to 0.5.
    """

    def __init__(self, transforms, p=0.5):
        t = [build_from_cfg(t, PIPELINES) for t in transforms]
        self.trans = _transforms.RandomApply(t, p=p)
        self.prob = p

    def __call__(self, results):
        return self.trans(results)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'prob = {self.prob}'
        return repr_str




@PIPELINES.register_module()
class Albu:
    """Albumentation augmentation.
    Adds custom transformations from Albumentations library.
    Please, visit `https://albumentations.readthedocs.io`
    to get more information.
    An example of ``transforms`` is as followed:
    .. code-block::
        [
            dict(
                type='ShiftScaleRotate',
                shift_limit=0.0625,
                scale_limit=0.0,
                rotate_limit=0,
                interpolation=1,
                p=0.5),
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(type='ChannelShuffle', p=0.1),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.1),
        ]
    Args:
        transforms (list[dict]): A list of albu transformations
        bbox_params (dict): Bbox_params for albumentation `Compose`
        keymap (dict): Contains {'input key':'albumentation-style key'}
        skip_img_without_anno (bool): Whether to skip the image if no ann left
            after aug
    """

    def __init__(self,
                 transforms):
        self.transforms = copy.deepcopy(transforms)
        self.aug = albumentations.Compose([self.albu_builder(t) for t in self.transforms])

    def albu_builder(self, cfg):
        assert isinstance(cfg, dict) and 'type' in cfg
        args = cfg.copy()

        obj_type = args.pop('type')
        if mmcv.is_str(obj_type):
            if albumentations is None:
                raise RuntimeError('albumentations is not installed')
            obj_cls = getattr(albumentations, obj_type)
        elif inspect.isclass(obj_type):
            obj_cls = obj_type
        else:
            raise TypeError(
                f'type must be a str or valid type, but got {type(obj_type)}')

        if 'transforms' in args:
            args['transforms'] = [
                self.albu_builder(transform)
                for transform in args['transforms']
            ]

        return obj_cls(**args)

    def __call__(self, results):
        if 'gt_semantic_seg' in results:
            tmp = self.aug(image=results['img'], mask=results['gt_semantic_seg'])
            results['img'] = tmp['image']
            results['gt_semantic_seg'] = tmp['mask']
        else:
            tmp = self.aug(image=results['img'])
            results['img'] = tmp['image']
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str


@PIPELINES.register_module()
class GenerateCutBox():
    def __init__(self, prop_range, n_boxes, crop_size, nomask=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.crop_size = crop_size
        self.nomask=nomask

    def generate_params(self):
        # Choose the proportion of each mask that should be above the threshold
        mask_props = np.random.uniform(self.prop_range[0], self.prop_range[1], size=(self.n_boxes))
        # Zeros will cause NaNs, so detect and suppres them
        zero_mask = mask_props == 0.0
        y_props = np.exp(np.random.uniform(low=0.0, high=1.0, size=(self.n_boxes)) * np.log(mask_props))
        x_props = mask_props / y_props
        fac = np.sqrt(1.0 / self.n_boxes)
        y_props *= fac
        x_props *= fac
        y_props[zero_mask] = 0
        x_props[zero_mask] = 0
        sizes = np.round(np.stack([y_props, x_props], axis=1) * np.array(self.crop_size)[ None, :])
        positions = np.round((np.array(self.crop_size) - sizes) * np.random.uniform(low=0.0, high=1.0, size=sizes.shape))
        rectangles = np.append(positions, positions + sizes, axis=1)
        masks = np.zeros(self.crop_size)
        for y0, x0, y1, x1 in rectangles:
            masks[int(y0):int(y1), int(x0):int(x1)] = 1 - masks[int(y0):int(y1), int(x0):int(x1)]
        if self.nomask:
            masks = np.ones_like(masks)
        return masks

    def __call__(self, results):
        cutmask = self.generate_params()
        results['cutmask'] = cutmask
        return results



@PIPELINES.register_module()
class SomeOfAugs(Compose):
    """Select N transforms to apply. Selected transforms will be called with `force_apply=True`.
    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.
    Args:
        transforms (list): list of transformations to compose.
        n (int): number of transforms to apply.
        replace (bool): Whether the sampled transforms are with or without replacement. Default: True.
        p (float): probability of applying selected transform. Default: 1.
    """

    def __init__(self, transforms, n, each_prob=None, replace=False, p=1, replay_mode=False):
        super(SomeOfAugs, self).__init__(transforms)
        self.n = n
        self.replace = replace
        assert sum(each_prob) == 1
        self.each_prob = each_prob
        self.replay_mode = replay_mode
        self.p = p

    def __call__(self, data):
        if self.replay_mode:
            for t in self.transforms:
                data = t(data)
                if data is None:
                    return None
            return data

        if self.each_prob and (random.random() < self.p):
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            transforms = random_state.choice(self.transforms, size=self.n, replace=self.replace, p=self.each_prob)
            for t in transforms:
                data = t(data)
                if data is None:
                    return None
            return data
        return data

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(transforms={self.transforms})'
        return repr_str







@PIPELINES.register_module()
class mosaic(object):
    # add by yuanjianlong
    def __init__(self, mix_pre_transforms, pre_transforms=None, prob=1., num_mosaic=3, img_scale=(512, 512),
                 center_ratio_range=(0.5, 1.5), pad_val=0, seg_pad_val=255):
        assert 0 <= prob and prob <= 1
        assert isinstance(img_scale, tuple)
        if pre_transforms is not None:
            self.pre_transforms = Compose(pre_transforms)
        else:
            self.pre_transforms = None
        self.mix_pre_transforms=Compose(mix_pre_transforms)
        self.num_mosaic = num_mosaic
        self.prob = prob
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def __call__(self, results):
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """
        ifmosaic = True if np.random.rand() < self.prob else False
        if ifmosaic:
            if self.pre_transforms is not None:
                results = self.pre_transforms(results)
            img_infos = results['img_infos']
            idxs = random.sample(list(range(len(img_infos))), self.num_mosaic)
            results["mix_results"] = []
            for idx in idxs:
                tmp = self.get_inputs(img_infos, idx, results['img_prefix'], results['seg_prefix'])
                tmp = self.mix_pre_transforms(tmp)
                results["mix_results"].append(tmp)
            results = self._mosaic_transform_img(results)
            results = self._mosaic_transform_seg(results)
            results.pop("mix_results")
        return results

    def get_inputs(self, img_infos, idx, img_prefix, seg_prefix):
        img_info = img_infos[idx]
        ann_info = img_infos[idx]['ann']
        return dict(img_info=img_info, ann_info=ann_info, img_prefix=img_prefix, seg_prefix=seg_prefix, seg_fields=[])

    def _mosaic_transform_img(self, results):
        """Mosaic transform function.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        if len(results['img'].shape) == 3:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2), 3),
                self.pad_val,
                dtype=results['img'].dtype)
        else:
            mosaic_img = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.pad_val,
                dtype=results['img'].dtype)

        # mosaic center x, y
        self.center_x = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[1])
        self.center_y = int(
            random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_position = (self.center_x, self.center_y)
        x1_p_m, y1_p_m, x2_p_m, y2_p_m = mosaic_img.shape[1], mosaic_img.shape[0], 0, 0
        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        for i, loc in enumerate(loc_strs):
            if loc == 'top_left':
                result_patch = copy.deepcopy(results)
            else:
                result_patch = copy.deepcopy(results['mix_results'][i - 1])

            img_i = result_patch['img']
            h_i, w_i = img_i.shape[:2]
            # keep_ratio resize
            scale_ratio_i = min(self.img_scale[0] / h_i,
                                self.img_scale[1] / w_i)
            img_i = mmcv.imresize(
                img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))
            # compute the combine parameters
            paste_coord, crop_coord = self._mosaic_combine(
                loc, center_position, img_i.shape[:2][::-1])
            x1_p, y1_p, x2_p, y2_p = paste_coord
            x1_c, y1_c, x2_c, y2_c = crop_coord
            x1_p_m, y1_p_m, x2_p_m, y2_p_m = min(x1_p_m, x1_p), min(y1_p_m, y1_p), max(x2_p_m, x2_p), max(y2_p_m, x2_p)
            # crop and paste image
            mosaic_img[y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]
        mosaic_img = mosaic_img[y1_p_m:y2_p_m, x1_p_m:x2_p_m]
        results['img'] = mosaic_img
        results['img_shape'] = mosaic_img.shape
        results['ori_shape'] = mosaic_img.shape

        return results

    def _mosaic_transform_seg(self, results):
        """Mosaic transform function for label annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """

        assert 'mix_results' in results
        for key in results.get('seg_fields', []):
            mosaic_seg = np.full(
                (int(self.img_scale[0] * 2), int(self.img_scale[1] * 2)),
                self.seg_pad_val,
                dtype=results[key].dtype)

            # mosaic center x, y
            center_position = (self.center_x, self.center_y)
            x1_p_m, y1_p_m, x2_p_m, y2_p_m = mosaic_seg.shape[1], mosaic_seg.shape[0], 0, 0
            loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
            for i, loc in enumerate(loc_strs):
                if loc == 'top_left':
                    result_patch = copy.deepcopy(results)
                else:
                    result_patch = copy.deepcopy(results['mix_results'][i - 1])

                gt_seg_i = result_patch[key]
                h_i, w_i = gt_seg_i.shape[:2]
                # keep_ratio resize
                scale_ratio_i = min(self.img_scale[0] / h_i,
                                    self.img_scale[1] / w_i)
                gt_seg_i = mmcv.imresize(
                    gt_seg_i,
                    (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)),
                    interpolation='nearest')

                # compute the combine parameters
                paste_coord, crop_coord = self._mosaic_combine(
                    loc, center_position, gt_seg_i.shape[:2][::-1])
                x1_p, y1_p, x2_p, y2_p = paste_coord
                x1_c, y1_c, x2_c, y2_c = crop_coord
                x1_p_m, y1_p_m, x2_p_m, y2_p_m = min(x1_p_m, x1_p), min(y1_p_m, y1_p), max(x2_p_m, x2_p), max(y2_p_m, x2_p)
                # crop and paste image
                mosaic_seg[y1_p:y2_p, x1_p:x2_p] = gt_seg_i[y1_c:y2_c,
                                                            x1_c:x2_c]
            mosaic_seg = mosaic_seg[y1_p_m:y2_p_m, x1_p_m:x2_p_m]
            results[key] = mosaic_seg

        return results

    def _mosaic_combine(self, loc, center_position_xy, img_shape_wh):
        """Calculate global coordinate of mosaic image and local coordinate of
        cropped sub-image.

        Args:
            loc (str): Index for the sub-image, loc in ('top_left',
              'top_right', 'bottom_left', 'bottom_right').
            center_position_xy (Sequence[float]): Mixing center for 4 images,
                (x, y).
            img_shape_wh (Sequence[int]): Width and height of sub-image

        Returns:
            tuple[tuple[float]]: Corresponding coordinate of pasting and
                cropping
                - paste_coord (tuple): paste corner coordinate in mosaic image.
                - crop_coord (tuple): crop corner coordinate in mosaic image.
        """

        assert loc in ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             center_position_xy[0], \
                             center_position_xy[1]
            crop_coord = img_shape_wh[0] - (x2 - x1), img_shape_wh[1] - (
                y2 - y1), img_shape_wh[0], img_shape_wh[1]

        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             max(center_position_xy[1] - img_shape_wh[1], 0), \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             center_position_xy[1]
            crop_coord = 0, img_shape_wh[1] - (y2 - y1), min(
                img_shape_wh[0], x2 - x1), img_shape_wh[1]

        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center_position_xy[0] - img_shape_wh[0], 0), \
                             center_position_xy[1], \
                             center_position_xy[0], \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = img_shape_wh[0] - (x2 - x1), 0, img_shape_wh[0], min(
                y2 - y1, img_shape_wh[1])

        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center_position_xy[0], \
                             center_position_xy[1], \
                             min(center_position_xy[0] + img_shape_wh[0],
                                 self.img_scale[1] * 2), \
                             min(self.img_scale[0] * 2, center_position_xy[1] +
                                 img_shape_wh[1])
            crop_coord = 0, 0, min(img_shape_wh[0],
                                   x2 - x1), min(y2 - y1, img_shape_wh[1])

        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'img_scale={self.img_scale}, '
        repr_str += f'center_ratio_range={self.center_ratio_range}, '
        repr_str += f'pad_val={self.pad_val}, '
        repr_str += f'seg_pad_val={self.pad_val})'
        return repr_str






@PIPELINES.register_module()
class copypaste(object):
    # add by yuanjianlong
    def __init__(self, mix_pre_transforms, pre_transforms=None, prob=1., num_imgs=1,
                 num_object=1, exclude=[0, 255]):
        assert 0 <= prob and prob <= 1
        if pre_transforms is not None:
            self.pre_transforms = Compose(pre_transforms)
        else:
            self.pre_transforms = None
        self.mix_pre_transforms=Compose(mix_pre_transforms)
        self.num_imgs = num_imgs
        self.num_object = num_object
        self.exclude = exclude
        self.prob = prob


    def __call__(self, results):
        """Call function to make a mosaic of image.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Result dict with mosaic transformed.
        """
        ifcopypaste = True if np.random.rand() < self.prob else False
        if ifcopypaste:
            if self.pre_transforms is not None:
                results = self.pre_transforms(results)
            img_infos = results['img_infos']
            num_imgs = min(self.num_imgs, len(img_infos))
            idxs = random.sample(list(range(len(img_infos))), num_imgs)
            objects = []
            for idx in idxs:
                tmp = self.get_inputs(img_infos, idx, results['img_prefix'], results['seg_prefix'])
                tmp = self.mix_pre_transforms(tmp)
                objects = self._select_object(tmp, objects)
            if len(objects) == 0:
                return results
            idxs = random.sample(list(range(len(objects))), min(self.num_object, len(objects)))

            img = results['img']
            mask = results['gt_semantic_seg']
            for idx in idxs:
                idx_obj = objects[idx]
                img, mask = self.paste(img, mask, idx_obj['img'], idx_obj['gt'], idx_obj['obj_mask'])
            results['img'] = img
            results['gt_semantic_seg'] = mask
        return results

    def get_inputs(self, img_infos, idx, img_prefix, seg_prefix):
        img_info = img_infos[idx]
        ann_info = img_infos[idx]['ann']
        return dict(img_info=img_info, ann_info=ann_info, img_prefix=img_prefix, seg_prefix=seg_prefix, seg_fields=[])

    def _select_object(self, results, objects):
        """Select some objects from the source results."""
        masks = results['gt_semantic_seg']
        img = results['img']
        idxs_all = list(set(np.unique(masks).tolist()) - set(self.exclude))
        for idx in idxs_all:
            obj_mask = (masks==idx).astype(np.uint8)
            objects.append(dict(img=img * np.expand_dims(obj_mask, axis=-1),
                                gt = masks * obj_mask,
                                obj_mask = obj_mask
                                ))

        return objects

    def paste(self, img, mask, obj_img, obj_gt, obj_mask):
        img = obj_img * np.expand_dims(obj_mask, axis=-1) + img * (1 -  np.expand_dims(obj_mask, axis=-1))
        mask = obj_gt * obj_mask + mask * (1 -  obj_mask)
        return img, mask
