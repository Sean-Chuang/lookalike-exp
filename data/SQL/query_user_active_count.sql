delete from z_seanchuang.tmp_user_active_count where dt='${dt}';

insert into z_seanchuang.tmp_user_active_count
select
    ad_id_plus,
    count(*) as n,
    '${dt}' as dt
from hive_ad.ml.ad_result_v3
where
    dt > date_format(date_add('day', -30, date('${dt}')), '%Y-%m-%d')
    and dt <= '${dt}'
group by 1;

