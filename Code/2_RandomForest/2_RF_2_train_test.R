####-------------------------------####
#~ source('/home/bisik/Practical/R/fun_0_loadLibrary.R')
source('../fun_0_loadLibrary.R')
####-------------------------------####
#~ source('/home/bisik/Practical/R/fun_2_2_trainRF.R')
source('fun_2_2_trainRF.R')

outputDir <- '/scratch-shared/edwin/test_rf_for_linda/train/'
dir.create(outputDir, showWarnings = F, recursive = T)

#-------train RF with tuned parameters on all available observations----------
#### all predictors ####
print('training: all predictors...')
#~ train_data <- vroom(paste0('/home/bisik/Practical/rf_input/bigTable_allpredictors_filtered_95.csv'),
#~                      show_col_types = F)
train_data <- vroom(paste0('/home/edwin/github/edwinkost/GLORIF1/example_dataset/Rhine_allpredictors.csv'),
                     show_col_types = F)

train_data_complete <- train_data


# skip some unreliable data
train_data <- train_data_complete
train_data <- train_data[which(train_data$grdc_no!=6335020),]

# select only the datetime after 1979 (1978 is for spinup)
train_data <- train_data[which(as.Date(train_data$datetime) >= as.Date("1979-01-01")),]

# and only until the year 2010 (2011 and after for validation)
train_data <- train_data[which(as.Date(train_data$datetime)  < as.Date("2011-01-01")),]



rf_input <- train_data %>% select(., -grdc_no, -cell_no_land, -datetime)
optimal_ranger <- trainRF(rf_input, num.trees=500, mtry=31, num.threads=48)

print('saving...')
saveRDS(optimal_ranger, paste0(outputDir,'trainedRF.rds'))                    
vi_df <- data.frame(names=names(optimal_ranger$variable.importance)) %>%
  mutate(importance=optimal_ranger$variable.importance)                     
write.csv(vi_df, paste0(outputDir,'varImportance.csv'), row.names=F)
