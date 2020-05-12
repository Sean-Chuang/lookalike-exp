# for train/test classifier usage
create table if not exists z_seanchuang.test_offline_training_set (
    cid bigint,
    ad_id_plus string,
    vimp1 tinyint,
    vimp2 tinyint,
    cv1 tinyint,
    cv2 tinyint
)
row format delimited fields terminated by '\t'
lines terminated by '\n'
stored as textfile
location 's3://smartad-dmp/warehouse/user/seanchuang/test_offline_training_set';