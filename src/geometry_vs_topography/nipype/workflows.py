import nibabel as ni

import nipype.interfaces.utility as util
import nipype.pipeline.engine as pe

from nipype_workbench_ext import cifti as wb_cifti
from nipype_workbench_ext import surface as wb_surface

from . import dual_regression as dual_reg

from warnings import warn


def init_tsnr(name='tsnr'):
    # this function produces an interface that can take one or more cifti input files,
    # computes tSNR for each, averages them acros inputs, estimates the mean tSNR
    # for each parcel specified by an input atlas, converts these to a csv file
    # of parcel tSNRs and returns that. It's designed to work equally with a 
    # scenario where tasks are spread across multiple scans or when tasks are in
    # a single scan.
    wf = pe.Workflow(name=name)

    inputnode = pe.Node(
        interface=util.IdentityInterface(
            fields=['in_files','atlas']),
        name='inputspec')

    joinWithinTask = pe.JoinNode(
        interface=util.IdentityInterface(
            fields=['in_file']),
        joinsource='directionsource',
        joinfield=['in_file'],
        name='joinwithintask')

    joinWithinSubject = pe.JoinNode(
        interface=util.IdentityInterface(
            fields=['in_file']),
        joinsource='tasksource',
        joinfield=['in_file'],
        name='joinwithinsubject')

    ciftiTSNR = pe.MapNode(
        interface=wb_cifti.Reduce(
            operation="TSNR"),
        iterfield=['in_file'],
        name='ciftitsnr')

    ciftiAverage = pe.Node(
        interface=wb_cifti.Average(),
        name='ciftiaverage')

    ciftiParcellate = pe.Node(
        interface=wb_cifti.Parcellate(),
        name='ciftiparcellate')

    ciftiToText = pe.Node(
        interface=wb_cifti.CiftiConvertText(),
        name='ciftiToText')

    outputspec = pe.Node(
        interface=util.IdentityInterface(
            fields=['out_file']),
        name='outputspec')
    
    wf.connect([
        (inputnode, joinWithinTask, [('in_files', 'in_file')]),
        (joinWithinTask, joinWithinSubject, [('in_file', 'in_file')]),
        (joinWithinSubject, ciftiTSNR, [(('in_file', mergelists), 'in_file')]),
        
        (ciftiTSNR, ciftiAverage, [('out_file', 'in_vars')]),
        
        (inputnode, ciftiParcellate, [('atlas', 'parcellation')]),
        (ciftiAverage, ciftiParcellate, [('out_file', 'in_file')]),

        (ciftiParcellate, ciftiToText, [('out_file', 'in_file')]),

        (ciftiToText, outputspec, [('out_file', 'out_file')])
    ])

    return wf


def init_hcp_dual_regression_wf(name='dualregressionwf', iterations=2, 
    surface_kernel=14, volume_kernel=14, zscorets=False, zscoremap=False):
    '''
    wf = init_hcp_dual_regression_wf(iterations=2, surface_kernel=14, volume_kernel=14)

    HCP inspired dual regression for surface data. This involves the following,
    1) construction of a alignment map, by dual regression of a low dimensional set of ICAs
       on one or more low dimensional ICA templates.
    2) construction of a weight map by combining the freesurfer areal distortion map with
       the alignment map. This weight map will be used for weighted dual regression to bias
       ICA timeseries estimation towards the areas that show the best alignment with the 
       group map.
    3) smoothing and contrast enhancement of the alignment map to refine the bias
    4) dual regression of timeseries data on a high dimensional ICA template using weight 
       map for timeseries estimation.
    5) repeated dual regression of timeseries data against individualized ICA maps produced
       by (4), using the surface area distortion map as a weight map without the alignment
       map (since both spatial maps and timeseries are now individualized). By default (5)
       is performed once, but can be repeated as many times as you like by adusting the
       iterations parameter.
    If a list of scans is provided to inputspec then spatial maps are computed for each and
    averaged on each iteration of dual regression (including alignment map construction).
    
    Glasser et al. (2016) Nature introduced several innovations for dual regression
    for surface based functional alignment. These are described in their supplemental
    methods section 2.3. This function implements a number of these innovations 
    (asterisks), while omitting a handful of other procedures they used
    - input timeseries unstructured noise variance normalization
    - timeseries concatenation
    - surface distortion weighting*
    - group alignment weighting*
    - iterative dual regression for individual differences enhancement*

    Unstructurd noise variance normalization was ommitted because it depends on 
    ICA-FIX outputs which may not always be available, and there's no sign of its
    use in their published code repo, despite the published methods description
    (see section 3.3 for this decription). If you'd like to learn more about this 
    method it appears they have an implementation here which may also be useful:
    https://github.com/Washington-University/HCPpipelines/blob/e165988f1fee786d950469665c9b7c7d111c0fb9/MSMAll/scripts/ComputeVN.m

    Temporal concatenation was omitted because in preliminary testing it appeared
    to produce noiser maps than averaging t-stats across runs. The latter approach 
    is used in the HCP_PTN1200 release. In particular look at the 
    subproc_DO_4_NodeTS.m file contained in their AWS repository:
    s3://hcp-openaccess/HCP_Resources/GroupAvg/HCP_PTN1200/scripts.tar.gz

    iterations - number of iterations of dual regression to use. First iterations is
                 against the group ICA maps, weighted by both alignment and distortion
                 maps. Subsequent iterations are against the last iterations mean
                 tstat map, therefore run/session specific, and only use the
                 distortion map. Note: default is set to Glasser's count, but I
                 find better test-retest reliability in HCP using 4-5. [Default = 2]

    *_kernel - smoothing kernels to use on alignment map. [Default = 14]

    zscorets - estimate standardized coefficients for timeseries GLMs [Default = False]

    zscoremap - estimate standardized coefficients for spatial GLMs [Default = False]
    
    The default kernel and iteration parameters above are the parameters used by 
    Glasser et al. (2016) Nature.

    Example usage :

        import glob
        from geometry_vs_topography.nipype import workflows as aligntools_wf

        HCP_ROOT = '/dartfs/rc/lab/D/DBIC/DBIC/archive/HCP'
        SID1 = '100307'

        groupICA = sorted(glob.glob(f'{HCP_ROOT}/HCP_Resources/GroupAvg/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d25.ica/melodic_IC.dscalar.nii'))[0]
        groupICAs = sorted(glob.glob(f'{HCP_ROOT}/HCP_Resources/GroupAvg/HCP_PTN1200/groupICA/groupICA_3T_HCP1200_MSMAll_d[12]5.ica/melodic_IC.dscalar.nii'))
        scans = sorted(glob.glob(f'{HCP_ROOT}/HCP1200/{SID1}/MNINonLinear/Results/rfMRI_REST*/rfMRI_REST[12]_[LR][RL]_Atlas_MSMAll_hp2000_clean.dtseries.nii'))
        surface_left=glob.glob(f'{HCP_ROOT}/HCP1200/{SID1}/MNINonLinear/fsaverage_LR32k/{SID1}.L.midthickness_MSMAll.32k_fs_LR.surf.gii')[0]
        surface_right=glob.glob(f'{HCP_ROOT}/HCP1200/{SID1}/MNINonLinear/fsaverage_LR32k/{SID1}.R.midthickness_MSMAll.32k_fs_LR.surf.gii')[0]
        roi_left=glob.glob(f'{HCP_ROOT}/HCP1200/{SID1}/MNINonLinear/fsaverage_LR32k/{SID1}.R.atlasroi.32k_fs_LR.shape.gii')[0]
        roi_right=glob.glob(f'{HCP_ROOT}/HCP1200/{SID1}/MNINonLinear/fsaverage_LR32k/{SID1}.R.atlasroi.32k_fs_LR.shape.gii')[0]

        wf = aligntools_wf.init_hcp_dual_regression_wf(iterations=5)

        wf.inputs.inputspec.func = scans                    # note this is a list
        wf.inputs.inputspec.group_ICAs = groupICA
        wf.inputs.inputspec.group_ICAs_low_d = groupICAs    # note this is a list. Glasser et al. use group ICAs of d=7..21
        wf.inputs.inputspec.roi_left = roi_left
        wf.inputs.inputspec.roi_right = roi_right
        wf.inputs.inputspec.surface_left = surface_left
        wf.inputs.inputspec.surface_right = surface_right

        wf.run()
    '''
    inputnode_dualregressionwf = pe.Node(
        interface=util.IdentityInterface(fields=[
            'func',
            'surface_left', 'surface_right',
            'roi_left', 'roi_right',
            'group_ICA','group_ICAs_low_d']),
        name='inputspec')

    wbSurfaceVertexAreaL = pe.Node(interface=wb_surface.SurfaceVertexAreas(),
        name="wbsurfacevertexareaL")

    wbSurfaceVertexAreaR = pe.Node(interface=wb_surface.SurfaceVertexAreas(),
        name="wbsurfacevertexareaR")

    wbCiftiCreateDenseScalar = pe.Node(interface=wb_cifti.CiftiCreateDenseScalar(),
        name="wbcifticreatedensescalar")

    meanVertexArea = pe.Node(interface=wb_cifti.CiftiStats(
            reduce='MEAN'),
        name="meanvertexarea")

    def getVAExpression(mean):
        return f'VA / {mean}'

    def getVAInVars(var):
        return [('VA', var)]

    normVertexArea = pe.Node(interface=wb_cifti.CiftiMath(),
        name="normvertexarea")

    alignmentMap = pe.Node(interface=dual_reg.estimateAlignmentMap(
            zscorets=zscorets,
            zscoremap=zscoremap),
        name="alignmentmap")

    smoothAlignmentMap = pe.Node(interface=wb_cifti.CiftiSmoothing(
            surface_kernel=surface_kernel,
            volume_kernel=volume_kernel),
        name="smoothalignmentmap")

    meanAlignment = pe.Node(interface=wb_cifti.CiftiStats(
            reduce='MEAN'),
        name="meanalignment")

    def getAlignmentMapExpression(mean):
        return f'(({mean}+weights-weightsm)*(({mean}+weights-weightsm)>0))^3'

    def getInVars(weights, weightsmooth):
        return [('weights', weights), ('weightsm', weightsmooth)]

    getinvars_node = pe.Node(util.Function(input_names=['weights', 'weightsmooth'],
                                                    output_names=['in_vars'],
                                                    function=getInVars),
                                        name='getinvars_node')

    enhanceAlignmentContrast = pe.Node(interface=wb_cifti.CiftiMath(),
        name="enhancealignmentcontrast")

    dualRegression = pe.Node(interface=dual_reg.iterativeDualRegression(
            iterations = iterations,
            zscorets = zscorets,
            zscoremap = zscoremap),
        name="dualregression")

    averageTStats = pe.Node(interface=wb_cifti.Average(),
        name="averagetstats")

    outputnode_dualregressionwf = pe.Node(
        interface=util.IdentityInterface(fields=[
            'timeseries','betas','tstats','zstats',
            'mean_tstat']),
        name='outputspec')

    dualRegressionWf = pe.Workflow(name=name)
    dualRegressionWf.connect([
        # assemble distortion correction file
        # This follows the approach described on lines 167-201 of 
        # https://github.com/Washington-University/HCPpipelines/blob/master/MSMAll/scripts/MSMAll.sh
        (inputnode_dualregressionwf, wbSurfaceVertexAreaL, [('surface_left', 'surface')]),
        (inputnode_dualregressionwf, wbSurfaceVertexAreaR, [('surface_right', 'surface')]),

        (inputnode_dualregressionwf, wbCiftiCreateDenseScalar, [('roi_left', 'left_roi'),
                                                                ('roi_right', 'right_roi')]),
        (wbSurfaceVertexAreaL, wbCiftiCreateDenseScalar, [('out_file', 'left_metric')]),
        (wbSurfaceVertexAreaR, wbCiftiCreateDenseScalar, [('out_file', 'right_metric')]),

        (wbCiftiCreateDenseScalar, normVertexArea, [(('out_file', getVAInVars), 'in_vars')]),
        (wbCiftiCreateDenseScalar, meanVertexArea, [('out_file', 'in_file')]),
        (meanVertexArea, normVertexArea, [(('value', getVAExpression), 'expression')]),

        # estimate alignment weights
        # this follows lines 90-118 of 
        # https://github.com/Washington-University/HCPpipelines/blob/e165988f1fee786d950469665c9b7c7d111c0fb9/MSMAll/scripts/MSMregression.m
        (inputnode_dualregressionwf, alignmentMap, [('func', 'in_files'),
                                                    ('group_ICAs_low_d', 'group_maps')]),
        (normVertexArea, alignmentMap, [('out_file', 'weights')]),

        (alignmentMap, smoothAlignmentMap, [('out_file', 'in_file')]),
        (inputnode_dualregressionwf, smoothAlignmentMap, [('surface_left', 'left_surface'),
                                                        ('surface_right', 'right_surface')]),

        (alignmentMap, getinvars_node, [('out_file', 'weights')]),
        (smoothAlignmentMap, getinvars_node, [('out_file', 'weightsmooth')]),
        (smoothAlignmentMap, meanAlignment, [('out_file', 'in_file')]),
        (meanAlignment, enhanceAlignmentContrast, [(('value', getAlignmentMapExpression), 'expression')]),
        (getinvars_node, enhanceAlignmentContrast, [('in_vars', 'in_vars')]),

        # do iterative weighted dual regression
        (inputnode_dualregressionwf, dualRegression, [('func', 'in_files')]),
        (enhanceAlignmentContrast, dualRegression, [('out_file', 'alignment_map')]),
        (normVertexArea, dualRegression, [('out_file', 'distortion_weights')]),
        (inputnode_dualregressionwf, dualRegression, [('group_ICA', 'group_map')]),

        # get average t-stat (redundant if len(inputspec.func) = 1)
        (dualRegression, averageTStats, [('tstat', 'in_vars')]),

        # output node
        (dualRegression, outputnode_dualregressionwf, [('timeseries', 'timeseries'),
                                                    ('beta', 'betas'),
                                                    ('tstat', 'tstats'),
                                                    ('zstat', 'zstats')]),
        (averageTStats, outputnode_dualregressionwf, [('out_file', 'mean_tstat')])
    ])

    return dualRegressionWf
