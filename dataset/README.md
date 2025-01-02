# Crunch1_autoimmune
## SpatialData object structure
Images
      'DAPI': DAPI image (validation and test tissue patches are removed)
      'DAPI_nuc': DAPI nucleus segmentation
      'HE_nuc_original': H&E nucleus segmentation on original image
      'HE_nuc_registered': H&E nucleus segmentation on registered image (registered to DAPI image)
      'HE_original': H&E original image
      'HE_registered': H&E registered image
      'group': Defining train(0)/validation(1)/test(2), No_transcript-train(4) tissue patches
      'group_HEspace': Defining train(0)/validation(1)/test(2), No_transcript-train(4) tissue patches on the H&E image
 
 Points
      'transcripts': DataFrame for each transcript (containing x,y,tissue patch,z_location,feature_name,transcript_id,qv,cell_id columns)
 
 Tables
       'anucleus':  AnnData contains .X, .layers['counts'], .obsm['spatial']
       'cell_id-group': AnnData only contains .obs DataFrame for mapping of cell_id to region.
with coordinate systems:
      'global', with elements:
        DAPI (Images), 'DAPI_nuc' (Images), 'HE_nuc_original' (Images), 'HE_nuc_registered'
            (Images), 'HE_original' (Images), 'HE_registered' (Images), 'group' (Images),6 'group_HEspace' (Images), 'transcripts' (Points)
      'scale_um_to_px', with elements: transcripts (Points)