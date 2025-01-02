# Crunch1_autoimmune
## Autoimmune Disease Machine Learning Challenge
* In Crunch 1, you will train an algorithm to predict spatial transcriptomics data (gene expression in each cell) from matched H&E images.
* Step 1: Identify nuclei in the H&E image
    Use the nucleus segmentation masks:
    H&E nucleus segmentation (HE_nuc_original): This mask identifies the location of nuclei in the original H&E image (i.e. HE_original).

* Step 2: Link gene expression to H&E images
    For each nucleus in the H&E image, use the anucleus file to get the corresponding gene expression profile (Y) for that nucleus.
    The anucleus file provides the gene expression data, where each row corresponds to a nucleus (cell) and each column corresponds to a gene.
    The nuclei IDs from the segmentation masks (e.g., from HE_nuc_original) will match the IDs used in the anucleus file.
