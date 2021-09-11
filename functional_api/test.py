import parent

import SimpleITK as sitk

import functional_api.functional as F

test_image_filename = '/playpen-raid1/tgreer/results/9279291/MR_SAG_3D_DESS/LEFT_KNEE/ENROLLMENT/image_preprocessed.nii.gz'


test_image = sitk.ReadImage(test_image_filename)
analysis_object = F.Analysis_Object()
FC_probmap, TC_probmap = analysis_object.segment(test_image)

FC_mesh, TC_mesh = analysis_object.extract_surface_mesh(FC_probmap, TC_probmap)

registration_result = analysis_object.register_image_to_atlas(test_image)


warped_FC_mesh = analysis_object.warp_mesh(FC_mesh, registration_result)
#warped_TC_mesh = analysis_object.warp_mesh(TC_mesh, registration_result)



