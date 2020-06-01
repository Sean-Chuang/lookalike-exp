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

# Add type & partidtion
drop table z_seanchuang.test_lal_offline_data;
create table if not exists z_seanchuang.test_lal_offline_data (
    type char(14),
    cid bigint,
    ad_id_plus string,
    vimp1 tinyint,
    vimp2 tinyint,
    cv1 tinyint,
    cv2 tinyint
)
partitioned by (dt string)
row format delimited fields terminated by '\t'
lines terminated by '\n'
stored as textfile
location 's3://smartad-dmp/warehouse/user/seanchuang/test_lal_offline_data';


create external table if not exists z_seanchuang.ad_deeplal_features (
ad_id_plus string,
luf string
)
partitioned by (type string, dt string)
row format delimited fields terminated by '\t'
lines terminated by '\n'
stored as textfile
location 's3://smartad-dmp/warehouse/user/seanchuang/ad_deeplal_features';


create table if not exists z_seanchuang.tmp_sampled_dmp_pixel (
  key string,
  uid string,
  url string,
  ts bigint
)
partitioned by (dt string)
stored as orc
location 's3://smartad-dmp/warehouse/user/seanchuang/tmp_sampled_dmp_pixel'
;


create table if not exists z_seanchuang.tmp_sampled_nginx_beacons (
  guid string,
  tag_id string,
  ts bigint
)
partitioned by (dt string)
stored as orc
location 's3://smartad-dmp/warehouse/user/seanchuang/tmp_sampled_nginx_beacons'
;

create table if not exists z_seanchuang.tmp_fasttext_training_set (
  sentence string
)
partitioned by (dt string)
row format delimited fields terminated by '\t'
lines terminated by '\n'
stored as textfile
location 's3://smartad-dmp/warehouse/user/seanchuang/tmp_fasttext_training_set'
;

create table if not exists z_seanchuang.tmp_daily_lal_campaign_info (
    advertiser_id bigint,
    owner_company string,
    campaign_id bigint,
    advertiser_name string,
    campaign_name string,
    click_action string,
    bid_type string,
    daily_budget double,
    monthly_budget double,
    targeting string,
    partners_url string,
    internal_version string
 )
partitioned by (dt string)
row format delimited fields terminated by '\t'
lines terminated by '\n'
stored as textfile
location 's3://smartad-dmp/warehouse/user/seanchuang/tmp_daily_lal_campaign_info';
