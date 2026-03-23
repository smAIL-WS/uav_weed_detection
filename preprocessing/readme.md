This file contains the categorization of the EWIS dataset as used in this paper. Each entry specifies the filename, its corresponding growth stage, and whether it belongs to the training or test split. Since the dataset published on Mendeley Data does not include predefined splits or growth stage labels, this categorization was created manually to ensure a stratified distribution across growth stages in both the training and test sets.

Note that the annotations for 10 additional testset images are not included in the Mendeley Data publication. These annotations are available in the repository under `annotations_additional_images/`.

uav_weed_detection/
└── raw_data/
    └── train/
        └── images/
            ├── BBCH_12/
                ├──img_20210713_B1_2_003.png 
                ├──img_20210713_B1_3_028.png  
                ├──img_20210713_B2_2_016.png  
                ├──img_20210713_B3_1_014.png  
                ├──img_20210824_D1_3_023.png  
                ├──img_20210827_D1_3_021.png
                ├──img_20210713_B1_2_022.png  
                ├──img_20210713_B2_1_008.png  
                ├──img_20210713_B2_3_007.png  
                ├──img_20210713_B3_1_020.png  
                ├──img_20210824_D1_3_028.png  
                ├──img_20210827_D1_3_028.png
                ├──img_20210713_B1_3_019.png  
                ├──img_20210713_B2_1_016.png  
                ├──img_20210713_B2_3_015.png  
                ├──img_20210824_D1_3_021.png  
                ├──img_20210827_D1_2_023.png
            ├── BBCH_13/
                ├──img_20210629_A1_1_002.png  
                ├──img_20210629_A1_1_029.png  
                ├──img_20210629_A1_2_027.png  
                ├──img_20210629_A1_3_031.png  
                ├──img_20210831_D1_3_021.png  
                ├──img_20210903_D1_3_014.png  
                ├──img_20210629_A1_1_010.png  
                ├──img_20210629_A1_2_006.png  
                ├──img_20210629_A1_3_005.png  
                ├──img_20210716_B3_1_011.png  
                ├──img_20210831_D1_3_028.png  
                ├──img_20210903_D1_3_021.png
                ├──img_20210629_A1_1_024.png  
                ├──img_20210629_A1_2_016.png  
                ├──img_20210629_A1_3_011.png  
                ├──img_20210831_D1_2_023.png  
                ├──img_20210903_D1_3_010.png  
                ├──img_20210903_D1_3_023.png
            ├── BBCH_14/
                ├──img_20210702_A1_2_003.png  
                ├──img_20210907_D1_2_023.png
            ├── BBCH_15/
                ├──img_20210705_A1_2_003.png  
                ├──img_20210810_C1_3_018.png  
                ├──img_20210810_C1_3_025.png
            ├── BBCH_16/
                ├──img_20210707_A1_1_002.png  
                ├──img_20210813_C1_3_001.png  
                ├──img_20210813_C2_1_013.png  
                ├──img_20210813_C2_2_020.png  
                ├──img_20210813_C3_1_024.png
                ├──img_20210708_A1_1_002.png  
                ├──img_20210813_C2_1_003.png  
                ├──img_20210813_C2_1_023.png  
                ├──img_20210813_C2_3_019.png  
                ├──img_20210914_D1_3_017.png
                ├──img_20210813_C1_1_002.png  
                ├──img_20210813_C2_1_007.png  
                ├──img_20210813_C2_2_005.png  
                ├──img_20210813_C3_1_005.png  
                ├──img_20210914_D2_2_012.png
            ├── BBCH_17/
                ├──img_20210917_D2_2_012.png
        └── annotations/
            └── same folders as /images/
                └── same file names but with .xml extension
    test/
        └── images/
            └── BBCH_13/
                └── Aholfing_20220525_Sorghum_008.png  
                └── Loew_20220505_Maize_018.png  
                └── Loew_20220509_Maize_020.png    
                └── Platte_20220516_Maize_006.png  
                └── Platte_20220516_Maize_041.png  
                └── Platte_20220516_Maize_051.png
                └── Aholfing_20220525_Sorghum_010.png  
                └── Loew_20220505_Maize_024.png  
                └── Loew_20220509_Maize_026.png    
                └── Platte_20220516_Maize_007.png  
                └── Platte_20220516_Maize_043.png
                └── Loew_20220505_Maize_006.png        
                └── Loew_20220509_Maize_006.png  
                └── Platte_20220516_Maize_004.png  
                └── Platte_20220516_Maize_008.png  
                └── Platte_20220516_Maize_047.png
                └── Loew_20220505_Maize_012.png        
                └── Loew_20220509_Maize_014.png  
                └── Platte_20220516_Maize_005.png  
                └── Platte_20220516_Maize_014.png  
                └── Platte_20220516_Maize_049.png
            └── BBCH_14/
                └── Loew_20220511_Maize_006.png  
                └── Loew_20220511_Maize_018.png  
                └── Platte_20220518_Maize_005.png  
                └── Platte_20220518_Maize_031.png
                └── Loew_20220511_Maize_012.png  
                └── Loew_20220511_Maize_024.png  
                └── Platte_20220518_Maize_021.png  
                └── Platte_20220602_Sorghum_004.png
            └── BBCH_15/
                └── Platte_20220520_Maize_005.png  
                └── Platte_20220520_Maize_016.png  
                └── Platte_20220520_Maize_018.png  
                └── Platte_20220520_Maize_022.png  
                └── Platte_20220520_Maize_033.png  
                └── Platte_20220520_Maize_048.png
            └── BBCH_16/
                └── Platte_20220523_Maize_003.png  
                └── Platte_20220523_Maize_005.png  
                └── Platte_20220523_Maize_008.png  
                └── Platte_20220523_Maize_028.png  
                └── Platte_20220523_Maize_044.png  
                └── Platte_20220523_Maize_048.png
            └── BBCH_17/
                └── Platte_20220608_Sorghum_006.png
        └── annotations/
            └── same folders as /images/
                └── same file names but with .xml extension