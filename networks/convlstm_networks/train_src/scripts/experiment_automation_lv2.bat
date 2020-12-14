@echo off

::KERAS_BACKEND=tensorflow
::id='sarh_tvalue20'
::id='sarh_tvalue40fixed'
::id='sarh_tvalue20repeat'
:: set id=windows_test
:: set id=int16_adagrad_crossentropy


::dataset=cv
set dataset=lm
::::dataSource='OpticalWithClouds'
::dataSource='SAR'
set dataSource=SAR
set model=BUnet4ConvLSTM
:: ==== EXTRACT PATCHES
:: call patches_extract.bat %dataset% %dataSource%

:: set id=lem_baseline_adam_focal_lessonedate1
:: call experiment_automation.bat %id% %model% %dataset% %dataSource%

:: set id=lem_baseline_adam_focal_lessonedate2
:: call experiment_automation.bat %id% %model% %dataset% %dataSource%

:: set id=lem_baseline_adam_focal_lessonedate3
:: call experiment_automation.bat %id% %model% %dataset% %dataSource%

:: set id=lem_baseline_adam_focal_lessonedate4
:: call experiment_automation.bat %id% %model% %dataset% %dataSource%

:: set id=lem_baseline_adam_focal_lessonedate5
:: call experiment_automation.bat %id% %model% %dataset% %dataSource%

set id=lem_regression3
call experiment_automation.bat %id% %model% %dataset% %dataSource%

:: set id=lem_baseline_adam_focal_alldates3
:: call experiment_automation.bat %id% %model% %dataset% %dataSource%

:: set id=lem_baseline_adam_focal_alldates4
:: call experiment_automation.bat %id% %model% %dataset% %dataSource%

:: set id=lem_baseline_adam_focal_alldates5
:: call experiment_automation.bat %id% %model% %dataset% %dataSource%

:: ===== USE MODEL
::. experiment_automation.sh $id 'BUnet4ConvLSTM_SkipLSTM' $dataset
::. experiment_automation.sh $id 'Unet3D' $dataset
::. experiment_automation.sh $id 'BUnet4ConvLSTM_64' $dataset  :: Unet5 uses 1 conv. in



::. experiment_automation.sh $id 'BUnet4ConvLSTM' $dataset $dataSource  :: Unet5 uses 1 conv. in
::. experiment_automation.sh $id 'ConvLSTM_seq2seq' $dataset  :: Unet5 uses 1 conv. in
::. experiment_automation.sh $id 'ConvLSTM_seq2seq_bi' $dataset
::. experiment_automation.sh $id 'BAtrousGAPConvLSTM' $dataset  :: gonna test balancing after replication
::. experiment_automation.sh $id 'DenseNetTimeDistributed_128x2' $dataset  :: Unet5 uses 1 conv. in

