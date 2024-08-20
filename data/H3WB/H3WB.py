# import os
# import sys
# import os.path as osp
# cwd = os.getcwd()
# sys.path.insert(0, osp.join('..', 'main'))
# sys.path.insert(0, osp.join('..', 'data'))
# sys.path.append(cwd)
# sys.path.append('main')
# # ------------------------

import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from common.utils.human_models import smpl_x
from common.utils.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, get_fitting_error_3D, resize_bbox
from common.utils.transforms import world2cam, cam2pixel, rigid_align
from common.utils.vis import render_mesh

from common.myutils.vis import DrawKeypPlat, reset_smplx_joint
plat = DrawKeypPlat()

class H3WB(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        self.img_dir = osp.join(cfg.data_dir, 'Human36M', 'images')
        self.annot_path = osp.join(cfg.data_dir, 'Human36M', 'annotations')
        self.action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']
        # H36M joint set
        self.joint_set = {'joint_num': 17,
                        'joints_name': ('Pelvis', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle', 'Torso', 'Neck', 'Head', 'Head_top', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'),
                        'flip_pairs': ( (1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13) ),
                        'eval_joint': (1, 2, 3, 4, 5, 6, 8, 10, 11, 12, 13, 14, 15, 16),
                        'regressor': np.load(osp.join(cfg.data_dir, 'Human36M', 'J_regressor_h36m_smplx.npy'))
                        }
        self.joint_set['root_joint_idx'] = self.joint_set['joints_name'].index('Pelvis')

        self.scene_video_name = None
        self.datalist = self.load_data()
        
    def get_subsampling_ratio(self):
        if self.data_split == 'train':
            return 5
        elif self.data_split == 'test':
            return 64
        else:
            assert 0, print('Unknown subset')

    def get_subject(self):
        if self.data_split == 'train':
            subject = [1,5,6,7,8]  
        elif self.data_split == 'test':
            subject = [9,11]
        else:
            assert 0, print("Unknown subset")

        return subject
    

    
    def load_data(self, video_frame_num=16):
        subject_list = self.get_subject()
        sampling_ratio = self.get_subsampling_ratio()
        
        # aggregate annotations from each subject
        db = COCO()
        cameras = {}
        joints = {}
        smplx_params = {}
        for subject in subject_list:
            # data load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_data.json'),'r') as f:
                annot = json.load(f)
            if len(db.dataset) == 0:
                for k,v in annot.items():
                    db.dataset[k] = v
            else:
                for k,v in annot.items():
                    db.dataset[k] += v
            # camera load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_camera.json'),'r') as f:
                cameras[str(subject)] = json.load(f)
            # joint coordinate load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_joint_3d.json'),'r') as f:
                joints[str(subject)] = json.load(f)
            # smplx parameter load
            with open(osp.join(self.annot_path, 'Human36M_subject' + str(subject) + '_SMPLX_NeuralAnnot.json'),'r') as f:
                smplx_params[str(subject)] = json.load(f)

        db.createIndex()

        video_split = []
        datalist = []
        for aid in db.anns.keys():
            ann = db.anns[aid]
            image_id = ann['image_id']
            img = db.loadImgs(image_id)[0]
            img_path = osp.join(self.img_dir, img['file_name'])
            img_shape = (img['height'], img['width'])
            
            # TODO
            video_name = img['file_name'].split('/')[0]
            

            # check subject and frame_idx
            frame_idx = img['frame_idx'];
            if frame_idx % sampling_ratio != 0:
                continue

            # smplx parameter
            subject = img['subject']; action_idx = img['action_idx']; subaction_idx = img['subaction_idx']; frame_idx = img['frame_idx']; cam_idx = img['cam_idx'];
            smplx_param = smplx_params[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)]

            # camera parameter
            cam_param = cameras[str(subject)][str(cam_idx)]
            R,t,f,c = np.array(cam_param['R'], dtype=np.float32), np.array(cam_param['t'], dtype=np.float32), np.array(cam_param['f'], dtype=np.float32), np.array(cam_param['c'], dtype=np.float32)
            cam_param = {'R': R, 't': t, 'focal': f, 'princpt': c}
            
            # only use frontal camera following previous works (HMR and SPIN)
            if self.data_split == 'test' and str(cam_idx) != '4':
                continue
                
            # project world coordinate to cam, image coordinate space
            joint_world = np.array(joints[str(subject)][str(action_idx)][str(subaction_idx)][str(frame_idx)], dtype=np.float32)
            joint_cam = world2cam(joint_world, R, t)
            joint_img = cam2pixel(joint_cam, f, c)[:,:2]
            joint_valid = np.ones((self.joint_set['joint_num'],1))
        
            bbox = process_bbox(np.array(ann['bbox']), img['width'], img['height'])
            if bbox is None: continue
            
            data_dict = {
                'img_path': img_path,
                'img_shape': img_shape,
                'bbox': bbox,
                'joint_img': joint_img,
                'joint_cam': joint_cam,
                'joint_valid': joint_valid,
                'smplx_param': smplx_param,
                'cam_param': cam_param}
            if self.scene_video_name == video_name:
                video_split.append(data_dict)
            else:
                self.scene_video_name = video_name
                video_split = []
                video_split.append(data_dict)
            if len(video_split) == video_frame_num:
                datalist.append(video_split)
                video_split = []

        return datalist
    
    def load_h3wb(task ='osxoptimVid',resultDirPath ='/home/u2021111446/CSPLAT/resultPlat/HPE/SA4D/wholebody3d/Result1/Human36m',testdata_folder= '/home/u2021111446/CsDataset/wholebody3d',actor='S6', vid_frame_num=16):
        import json 
        with open(testdata_folder+'/datasets/json/RGBto3D_train_img_cs.json' , 'r') as jsonfile:
            RGBto3D_train_img_cs = json.load(jsonfile)
        datalist = []
        videos = RGBto3D_train_img_cs[actor]
        for video in videos.keys():
            img_list, wh3d_list, bbox_list = [], [], []
            imgs_dic = RGBto3D_train_img_cs[actor][video]
            imgs = sorted(imgs_dic)
            for img in imgs:
                img_path = RGBto3D_train_img_cs[actor][video][img]['img_path']
                wholebody3d = RGBto3D_train_img_cs[actor][video][img]['3d_wholebody_pose']
                bbox = RGBto3D_train_img_cs[actor][video][img]['bbox']
                img_list.append(img_path)
                wh3d_list.append(wholebody3d)
                bbox_list.append(bbox) 
                if len(img_list) == vid_frame_num:
                    vid_name = img_path.split('frame_')[0] + 'vidclip'+ img_list[0].split('frame_')[1][:4] + '_' + img_list[-1].split('frame_')[1][:4]
                    vid_name = vid_name.replace('Images', 'Video')
                    out_seq_path = os.path.join(resultDirPath, vid_name.split('vidclip')[0] + task + '_' + vid_name.split('/')[-1] + '.pkl')
                    datalist.append({vid_name: img_list, 'out_vid_path': out_seq_path, 'wh3d_vid': wh3d_list, 'bbox_vid': bbox_list})
                    img_list, wh3d_list, bbox_list = [], [], []
                pass
        # CSPLAT/resultPlat/HPE/SA4D/wholebody3d/Result1/Human36m/
        # S6/Video/Smoking 1.60457274/osxoptimVid_vidclip0000_0225.pkl 
            
                
        return datalist
    
    def smplx_joints_img2bboxes(self, human_model_param, cam_param, img_shape, img2bb_trans,do_flip):
        rotation_valid = np.ones((smpl_x.orig_joint_num), dtype=np.float32)
        coord_valid = np.ones((smpl_x.joint_num), dtype=np.float32)

        root_pose, body_pose, shape, trans = human_model_param['root_pose'], human_model_param['body_pose'], \
                                             human_model_param['shape'], human_model_param['trans']
        if 'lhand_pose' in human_model_param and human_model_param['lhand_valid']:
            lhand_pose = human_model_param['lhand_pose']
        else:
            lhand_pose = np.zeros((3 * len(smpl_x.orig_joint_part['lhand'])), dtype=np.float32)
            rotation_valid[smpl_x.orig_joint_part['lhand']] = 0
            coord_valid[smpl_x.joint_part['lhand']] = 0
        if 'rhand_pose' in human_model_param and human_model_param['rhand_valid']:
            rhand_pose = human_model_param['rhand_pose']
        else:
            rhand_pose = np.zeros((3 * len(smpl_x.orig_joint_part['rhand'])), dtype=np.float32)
            rotation_valid[smpl_x.orig_joint_part['rhand']] = 0
            coord_valid[smpl_x.joint_part['rhand']] = 0
        if 'jaw_pose' in human_model_param and 'expr' in human_model_param and human_model_param['face_valid']:
            jaw_pose = human_model_param['jaw_pose']
            expr = human_model_param['expr']
            expr_valid = True
        else:
            jaw_pose = np.zeros((3), dtype=np.float32)
            expr = np.zeros((smpl_x.expr_code_dim), dtype=np.float32)
            rotation_valid[smpl_x.orig_joint_part['face']] = 0
            coord_valid[smpl_x.joint_part['face']] = 0
            expr_valid = False
        if 'gender' in human_model_param:
            gender = human_model_param['gender']
        else:
            gender = 'neutral'
        root_pose = torch.FloatTensor(root_pose).view(1, 3)  # (1,3)
        body_pose = torch.FloatTensor(body_pose).view(-1, 3)  # (21,3)
        lhand_pose = torch.FloatTensor(lhand_pose).view(-1, 3)  # (15,3)
        rhand_pose = torch.FloatTensor(rhand_pose).view(-1, 3)  # (15,3)
        jaw_pose = torch.FloatTensor(jaw_pose).view(-1, 3)  # (1,3)
        shape = torch.FloatTensor(shape).view(1, -1)  # SMPLX shape parameter
        expr = torch.FloatTensor(expr).view(1, -1)  # SMPLX expression parameter
        trans = torch.FloatTensor(trans).view(1, -1)  # translation vector

        # apply camera extrinsic (rotation)
        # merge root pose and camera rotation
        if 'R' in cam_param:
            R = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3)
            root_pose = root_pose.numpy()
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(R, root_pose))
            root_pose = torch.from_numpy(root_pose).view(1, 3)

        # get mesh and joint coordinates
        zero_pose = torch.zeros((1, 3)).float()  # eye poses
        with torch.no_grad():
            output = smpl_x.layer[gender](betas=shape, body_pose=body_pose.view(1, -1), global_orient=root_pose,
                                          transl=trans, left_hand_pose=lhand_pose.view(1, -1),
                                          right_hand_pose=rhand_pose.view(1, -1), jaw_pose=jaw_pose.view(1, -1),
                                          leye_pose=zero_pose, reye_pose=zero_pose, expression=expr)
        # mesh_cam = output.vertices[0].numpy()
        joint_cam = output.joints[0].numpy()[smpl_x.joint_idx, :]

        # apply camera exrinsic (translation)
        # compenstate rotation (translation from origin to root joint was not cancled)
        if 'R' in cam_param and 't' in cam_param:
            R, t = np.array(cam_param['R'], dtype=np.float32).reshape(3, 3), np.array(cam_param['t'],
                                                                                      dtype=np.float32).reshape(1, 3)
            root_cam = joint_cam[smpl_x.root_joint_idx, None, :]
            joint_cam = joint_cam - root_cam + np.dot(R, root_cam.transpose(1, 0)).transpose(1, 0) + t
        
        # joint coordinates
        joint_img = cam2pixel(joint_cam, cam_param['focal'], cam_param['princpt'])
        joint_cam = joint_cam - joint_cam[smpl_x.root_joint_idx, None, :]  # root-relative
        joint_cam[smpl_x.joint_part['lhand'], :] = joint_cam[smpl_x.joint_part['lhand'], :] - joint_cam[
                                                                                              smpl_x.lwrist_idx, None,
                                                                                              :]  # left hand root-relative
        joint_cam[smpl_x.joint_part['rhand'], :] = joint_cam[smpl_x.joint_part['rhand'], :] - joint_cam[
                                                                                              smpl_x.rwrist_idx, None,
                                                                                              :]  # right hand root-relative
        joint_cam[smpl_x.joint_part['face'], :] = joint_cam[smpl_x.joint_part['face'], :] - joint_cam[smpl_x.neck_idx,
                                                                                            None,
                                                                                            :]  # face root-relative
        joint_img[smpl_x.joint_part['body'], 2] = (joint_cam[smpl_x.joint_part['body'], 2].copy() / (
                    cfg.body_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # body depth discretize
        joint_img[smpl_x.joint_part['lhand'], 2] = (joint_cam[smpl_x.joint_part['lhand'], 2].copy() / (
                    cfg.hand_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # left hand depth discretize
        joint_img[smpl_x.joint_part['rhand'], 2] = (joint_cam[smpl_x.joint_part['rhand'], 2].copy() / (
                    cfg.hand_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # right hand depth discretize
        joint_img[smpl_x.joint_part['face'], 2] = (joint_cam[smpl_x.joint_part['face'], 2].copy() / (
                    cfg.face_3d_size / 2) + 1) / 2. * cfg.output_hm_shape[0]  # face depth discretize

        joint_img, _ = reset_smplx_joint(joint_img)
        lhand_joint_img = joint_img[25:46]
        rhand_joint_img = joint_img[46:67]
        face_joint_img = joint_img[67:]
        
        # rough bboxes
        lhand_bbox = [min(lhand_joint_img[:, 0]), min(lhand_joint_img[:, 1]),
                max(lhand_joint_img[:, 0]), max(lhand_joint_img[:, 1])]
        lhand_bbox = resize_bbox(lhand_bbox, scale=1.2)
        lhand_bbox = self.process_hand_face_bbox(bbox=np.array(lhand_bbox),
                                                 do_flip = do_flip,
                                                img_shape = img_shape, 
                                                img2bb_trans = img2bb_trans)
        
        rhand_bbox = [min(rhand_joint_img[:, 0]), min(rhand_joint_img[:, 1]),
                max(rhand_joint_img[:, 0]), max(rhand_joint_img[:, 1])]
        rhand_bbox = resize_bbox(rhand_bbox, scale=1.2)
        rhand_bbox = self.process_hand_face_bbox(bbox=np.array(rhand_bbox),
                                                 do_flip = do_flip,
                                                img_shape = img_shape, 
                                                img2bb_trans = img2bb_trans)
        
        face_bbox = [min(face_joint_img[:, 0]), min(face_joint_img[:, 1]),
                max(face_joint_img[:, 0]), max(face_joint_img[:, 1])]
        face_bbox = resize_bbox(face_bbox, scale=1.2)
        face_bbox = self.process_hand_face_bbox(bbox=np.array(face_bbox),
                                                do_flip = do_flip,
                                                img_shape = img_shape, 
                                                img2bb_trans = img2bb_trans)
                    
        return lhand_bbox, rhand_bbox, face_bbox 
    
    def process_hand_face_bbox(self, bbox, do_flip, img_shape, img2bb_trans, bbox_type='xyhw'):
        if bbox is None:
            bbox = np.array([0, 0, 1, 1], dtype=np.float32).reshape(2, 2)  # dummy value
            bbox_valid = float(False)  # dummy value
        else:
            # reshape to top-left (x,y) and bottom-right (x,y)
            bbox = bbox.reshape(2, 2)

            # flip augmentation
            if do_flip:
                bbox[:, 0] = img_shape[1] - bbox[:, 0] - 1
                bbox[0, 0], bbox[1, 0] = bbox[1, 0].copy(), bbox[0, 0].copy()  # xmin <-> xmax swap

            # make four points of the bbox
            bbox = bbox.reshape(4).tolist()
            xmin, ymin, xmax, ymax = bbox
            bbox = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=np.float32).reshape(4, 2)

            # affine transformation (crop, rotation, scale)
            bbox_xy1 = np.concatenate((bbox, np.ones_like(bbox[:, :1])), 1)
            bbox = np.dot(img2bb_trans, bbox_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            bbox[:, 0] = bbox[:, 0] / cfg.input_img_shape[1] * cfg.output_hm_shape[2]
            bbox[:, 1] = bbox[:, 1] / cfg.input_img_shape[0] * cfg.output_hm_shape[1]

            # make box a rectangle without rotation
            xmin = np.min(bbox[:, 0]);
            xmax = np.max(bbox[:, 0]);
            ymin = np.min(bbox[:, 1]);
            ymax = np.max(bbox[:, 1]);
            bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)

            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            x_center, y_center = (x1+x2)/2.0, (y1+y2)/2.0
            x_size, y_size = x2-x1, y2-y1
            bbox = np.array([[x_center, y_center], [x_size, y_size]])

        return bbox

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        data_video_split = copy.deepcopy(self.datalist[idx])
        # video_data2model = []
        concat_inputs = {'vid': []}
        concat_targets = {'joint_img': [], 'smplx_joint_img': [], 
                          'joint_cam': [], 'smplx_joint_cam': [], 
                          'smplx_pose': [], 'smplx_shape': [], 'smplx_expr': [], 
                          'lhand_bbox_center': [], 'lhand_bbox_size': [], 
                          'rhand_bbox_center': [], 'rhand_bbox_size': [], 
                          'face_bbox_center': [], 'face_bbox_size': []}
        concat_meta_info = {'joint_valid': [], 'joint_trunc': [], 
                            'smplx_joint_valid': [], 'smplx_joint_trunc': [], 
                            'smplx_pose_valid': [], 'smplx_shape_valid': [], 
                            'smplx_expr_valid': [], 'is_3D': [], 
                            'lhand_bbox_valid': [], 'rhand_bbox_valid': [], 
                            'face_bbox_valid': []}
        

        for data in data_video_split:
            img_path, img_shape, bbox, cam_param = data['img_path'], data['img_shape'], data['bbox'], data['cam_param']
            
            # img
            img = load_img(img_path)
            img, img2bb_trans, bb2img_trans, rot, do_flip = augmentation(img, bbox, self.data_split)
            img = self.transform(img.astype(np.float32))/255.
            
            
            if self.data_split == 'train':
                # h36m gt
                joint_cam = data['joint_cam']
                joint_cam = (joint_cam - joint_cam[self.joint_set['root_joint_idx'],None,:]) / 1000 # root-relative. milimeter to meter.
                joint_img = data['joint_img']
                joint_img = np.concatenate((joint_img[:,:2], joint_cam[:,2:]),1) # x, y, depth
                joint_img[:,2] = (joint_img[:,2] / (cfg.body_3d_size / 2) + 1)/2. * cfg.output_hm_shape[0] # discretize depth
                joint_img, joint_cam, joint_valid, joint_trunc = process_db_coord(joint_img, joint_cam, data['joint_valid'], do_flip, img_shape, self.joint_set['flip_pairs'], img2bb_trans, rot, self.joint_set['joints_name'], smpl_x.joints_name)
                
                # smplx coordinates and parameters
                smplx_param = data['smplx_param']
                cam_param['t'] /= 1000 # milimeter to meter
                smplx_joint_img, smplx_joint_cam, smplx_joint_trunc, smplx_pose, smplx_shape, smplx_expr, smplx_pose_valid, smplx_joint_valid, smplx_expr_valid, smplx_mesh_cam_orig = process_human_model_output(smplx_param, cam_param, do_flip, img_shape, img2bb_trans, rot, 'smplx')

                # dummy hand/face bbox
                lhand_bbox, rhand_bbox, face_bbox = self.smplx_joints_img2bboxes(smplx_param, cam_param, img_shape, img2bb_trans, do_flip)
            
                
                """
                # for debug
                _tmp = joint_img.copy()
                _tmp[:,0] = _tmp[:,0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
                _tmp[:,1] = _tmp[:,1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
                _img = img.numpy().transpose(1,2,0)[:,:,::-1] * 255
                _img = vis_keypoints(_img, _tmp)
                cv2.imwrite('h36m_' + str(idx) + '.jpg', _img)
                """

                # SMPLX pose parameter validity
                for name in ('L_Ankle', 'R_Ankle', 'L_Wrist', 'R_Wrist'):
                    smplx_pose_valid[smpl_x.orig_joints_name.index(name)] = 0
                smplx_pose_valid = np.tile(smplx_pose_valid[:,None], (1,3)).reshape(-1)
                # SMPLX joint coordinate validity
                for name in ('L_Big_toe', 'L_Small_toe', 'L_Heel', 'R_Big_toe', 'R_Small_toe', 'R_Heel'):
                    smplx_joint_valid[smpl_x.joints_name.index(name)] = 0
                smplx_joint_valid = smplx_joint_valid[:,None]
                smplx_joint_trunc = smplx_joint_valid * smplx_joint_trunc
                smplx_shape_valid = True

                # # dummy hand/face bbox
                # dummy_center = np.zeros((2), dtype=np.float32)
                # dummy_size = np.zeros((2), dtype=np.float32)

                inputs = {'img': img}
                targets = {'joint_img': joint_img, 'smplx_joint_img': smplx_joint_img, 
                       'joint_cam': joint_cam, 'smplx_joint_cam': smplx_joint_cam, 
                       'smplx_pose': smplx_pose, 'smplx_shape': smplx_shape, 
                       'smplx_expr': smplx_expr, 'lhand_bbox_center': lhand_bbox[0], 
                       'lhand_bbox_size': lhand_bbox[1], 'rhand_bbox_center': rhand_bbox[0], 
                       'rhand_bbox_size': rhand_bbox[1], 'face_bbox_center': face_bbox[0], 
                       'face_bbox_size': face_bbox[1]}
                meta_info = {'joint_valid': joint_valid, 'joint_trunc': joint_trunc, 
                            'smplx_joint_valid': smplx_joint_valid, 'smplx_joint_trunc': smplx_joint_trunc, 
                            'smplx_pose_valid': smplx_pose_valid, 'smplx_shape_valid': float(smplx_shape_valid), 
                            'smplx_expr_valid': float(smplx_expr_valid), 'is_3D': float(True), 
                            'lhand_bbox_valid': float(True), 'rhand_bbox_valid': float(True), 
                            'face_bbox_valid': float(True)}
                # video_data2model.append([inputs, targets, meta_info])
                concat_inputs['vid'].append(np.expand_dims(inputs['img'], axis=0))
                for key in targets.keys():
                    # if isinstance(targets[key], np.ndarray) and targets[key].shape[0] != 1:
                    #     concat_targets[key].append(np.expand_dims(targets[key], axis=0))
                    # else:
                        concat_targets[key].append(targets[key])
                for key in meta_info.keys():
                    # if isinstance(meta_info[key], np.ndarray) and meta_info[key].shape[0] != 1:
                    #     concat_meta_info[key].append(np.expand_dims(meta_info[key], axis=0))
                    # else:
                        concat_meta_info[key].append(meta_info[key])
                pass
                
            else:
                inputs = {'img': img}
                targets = {}
                meta_info = {}

                # video_data2model.append([inputs, targets, meta_info])
                concat_inputs['vid'].append(np.expand_dims(inputs['img'], axis=0))
                

        # return video_data2model
        concat_inputs['vid'] = np.concatenate(concat_inputs['vid'], axis=0)
        if self.data_split == 'train':
            for key in concat_targets.keys():
                if isinstance(concat_targets[key], np.ndarray):
                    # if concat_targets[key][0].shape[0] != 1:
                    #     concat_targets[key].append(np.expand_dims(targets[key], axis=0))
                    concat_targets[key] = np.concatenate(concat_targets[key], axis=0)
                else:
                    concat_targets[key] = np.array(concat_targets[key])
            for key in concat_meta_info.keys():
                if isinstance(concat_meta_info[key], np.ndarray):
                    concat_meta_info[key] = np.concatenate(concat_meta_info[key], axis=0)
                else:
                    concat_meta_info[key] = np.array(concat_meta_info[key])
        else:
            concat_targets = {}
            concat_meta_info = {}
            

        return concat_inputs, concat_targets, concat_meta_info
    
    

    def evaluate(self, outs, cur_sample_idx):

        annots = self.datalist
        sample_num = len(outs)
        eval_result = {'mpjpe': [], 'pa_mpjpe': []}
        for n in range(sample_num):
            video_annot = annots[cur_sample_idx + n]
            video_out = outs[n]
            video_mpjpe = []
            video_pa_mpjpe = []
            for annot, out_smplx_mesh_cam in zip(video_annot, video_out['smplx_mesh_cam']):
                # h36m joint from gt mesh
                joint_gt = annot['joint_cam'] 
                joint_gt = joint_gt - joint_gt[self.joint_set['root_joint_idx'],None] # root-relative 
                joint_gt = joint_gt[self.joint_set['eval_joint'],:] 
                
                # h36m joint from param mesh
                mesh_out = out_smplx_mesh_cam * 1000 # meter to milimeter
                joint_out = np.dot(self.joint_set['regressor'], mesh_out) # meter to milimeter
                joint_out = joint_out - joint_out[self.joint_set['root_joint_idx'],None] # root-relative
                joint_out = joint_out[self.joint_set['eval_joint'],:]
                joint_out_aligned = rigid_align(joint_out, joint_gt)
                video_mpjpe.append(np.sqrt(np.sum((joint_out - joint_gt)**2,1)).mean())
                video_pa_mpjpe.append(np.sqrt(np.sum((joint_out_aligned - joint_gt)**2,1)).mean())


                
                
                # # render mesh
                # bbox = annot['bbox']
                # vis_img = load_img(annot['img_path'])[:,:,::-1]
                # focal = [cfg.focal[0] / cfg.input_body_shape[1] * bbox[2], cfg.focal[1] / cfg.input_body_shape[0] * bbox[3]]
                # princpt = [cfg.princpt[0] / cfg.input_body_shape[1] * bbox[2] + bbox[0], cfg.princpt[1] / cfg.input_body_shape[0] * bbox[3] + bbox[1]]
                # vis_img = render_mesh(vis_img, out_smplx_mesh_cam, smpl_x.face, {'focal': focal, 'princpt': princpt})


            eval_result['mpjpe'].append(np.array(video_mpjpe).mean())
            eval_result['pa_mpjpe'].append(np.array(video_pa_mpjpe).mean())


            vis = False
            if vis:
                from common.utils.vis import vis_keypoints, vis_mesh, save_obj
                filename = annot['img_path'].split('/')[-1][:-4]

                img = load_img(annot['img_path'])[:,:,::-1]
                img = vis_mesh(img, mesh_out_img, 0.5)
                cv2.imwrite(filename + '.jpg', img)
                save_obj(mesh_out, smpl_x.face, filename + '.obj')

        return eval_result

    def print_eval_result(self, eval_result):
        print('MPJPE: %.2f mm' % np.mean(eval_result['mpjpe']))
        print('PA MPJPE: %.2f mm' % np.mean(eval_result['pa_mpjpe']))



# if __name__ == '__main__':
#     from torchvision import transforms
#     # trainset3d_loader.append(eval(cfg.trainset_3d[i])(transforms.ToTensor(), "train"))
#     # cfg.data_dir = '/home/u2021111446/CsDataset'
#     trainset3d_loader = Human36M(transforms.ToTensor(), "train")
#     for data in trainset3d_loader:
#         print(data)
#         break
    

