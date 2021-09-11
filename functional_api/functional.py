import mn_oai_pipeline
import SimpleITK as sitk
import numpy as np
from registration.registers import NiftyReg, AVSMReg
from segmentation.segmenter import Segmenter3DInPatchClassWise
import os
import shape_analysis.cartilage_shape_processing
class Analysis_Object:

    """Contains the configuration and loaded weights of the 
    pipeline, but _not_ state related to running any given 
    image through the pipeline"""
    def __init__(self):
        """For now, not configurable
        
        Configuration TODO:
        flag for nifti vs avsm
        """
        
        # Configuration loaded from marc's code
        PARAMS = mn_oai_pipeline.PARAMS
        for key, value in mn_oai_pipeline.DEFAULT.items():
            PARAMS[key] = value

        # values in PARAMS must be absolute paths and contain no spaces.
        PARAMS["avsm_directory"] = os.path.abspath("dependencies/easyreg")
        PARAMS["atlas_image"] = os.path.abspath("atlas/atlas_60_LEFT_baseline_NMI/atlas.nii.gz")
        self.PARAMS = PARAMS

        #Configure registration algorithm
        use_nifty = False
        niftyreg_path = PARAMS['nifty_reg_directory']
        avsm_path = PARAMS["avsm_directory"]
        avsm_path = avsm_path + '/demo'


        self.register = NiftyReg(niftyreg_path) if use_nifty else AVSMReg(avsm_path=avsm_path,python_executable=PARAMS['python_executable'])
        """
        affine_config = dict(smooth_moving=-1, smooth_ref=-1,
                             max_iterations=10,
                             pv=30, pi=30,
                             num_threads=30)
        bspline_config = dict(
            max_iterations=300,
            # num_levels=3, performed_levels=3,
            smooth_moving=-1, smooth_ref=0,
            sx=4, sy=4, sz=4,
            num_threads=32,
            be=0.1,  # bending energy, second order derivative of deformations (0.01)
        )
        """
        # Configure segmentation algorithm
        ckpoint_folder = "./segmentation/ckpoints/UNet_bias_Nifti_rescaled_LEFT_train1_patch_128_128_32_batch_4_sample_0.01-0.02_BCEWithLogitsLoss_lr_0.001/01272019_212723"
        segmenter_config = dict(
            ckpoint_path=os.path.join(ckpoint_folder, "model_best.pth.tar"),
            training_config_file=os.path.join(ckpoint_folder, "train_config.json"),
            device="cuda",
            batch_size=4,
            overlap_size=(16, 16, 8),
            output_prob=True,
            output_itk=True,
        )
        self.segmenter = Segmenter3DInPatchClassWise(mode="pred", config=segmenter_config)

    def segment(self, preprocessed_image):
        FC_probmap, TC_probmap = self.segmenter.segment(preprocessed_image, if_output_prob_map=True, if_output_itk=True)
        return (FC_probmap, TC_probmap)

    def extract_surface_mesh(self, FC_probmap, TC_probmap):
        FC_prob = np.swapaxes(sitk.GetArrayFromImage(FC_probmap), 0, 2).astype(float)
        TC_prob = np.swapaxes(sitk.GetArrayFromImage(TC_probmap), 0, 2).astype(float)
        transform = None #shape_analysis.cartilage_shape_processing.get_voxel_to_world_transform_nifti(FC_probmap)
        return shape_analysis.cartilage_shape_processing.get_cartilage_surface_mesh_from_segmentation_array(FC_prob, TC_prob,
                                                                  spacing=FC_probmap.GetSpacing(),
                                                                  thickness=True,
                                                                  prob=True,
                                                                  transform=transform)


    def register_image_to_atlas(self, unsegmented_image):
        # as a first pass, we will write to temporary files
        import random
        import pathlib
        import os
        import nibabel
        
        tmpdir = os.path.abspath(str(random.random())[4:])
        try:
            os.mkdir(tmpdir)

            unsegmented_image_path = os.path.join(tmpdir, "image_preprocessed.nii.gz")
            sitk.WriteImage(unsegmented_image, unsegmented_image_path)
            
            fake_oai_image = lambda: None
            fake_oai_image.inv_transform_to_atlas = os.path.join(tmpdir, "image_preprocessed_atlas_inv_phi.nii.gz")
            self.register.register_image(self.PARAMS["atlas_image"], unsegmented_image_path,
                                             lmoving_path=None, ltarget_path=None,
                                             gpu_id=0,oai_image=fake_oai_image)
            return nibabel.load(fake_oai_image.inv_transform_to_atlas)
            
            
        finally:
            pass
            #os.rmdir(tmpdir)

    def warp_mesh(self, mesh, displacement_field):
        pass

