delete from z_seanchuang.test_offline_training_set;

insert into z_seanchuang.test_offline_training_set
with u1 as (
    select
        id as campaign_id,
        ad_id as ad_id_plus,
        max(cast(postback as tinyint)) as cv
    from ml.ad_result_v3 ar
    where
        dt between '2020-04-10'
        and '2020-04-25'
        and (opt_tg = 2 or bid_t = 3)
    group by
        1,
        2
),
u2 as (
    select
        id as campaign_id,
        ad_id as ad_id_plus,
        max(cast(postback as tinyint)) as cv
    from ml.ad_result_v3 ar
    where
        dt between '2020-04-26'
        and '2020-05-11'
        and (opt_tg = 2 or bid_t = 3)
    group by
        1,
        2
),
v1_cv_raw as (
    select
        campaign_id,
        slice(shuffle(array_agg(ad_id_plus)), 1, 100) as ad_ids
    from u1
    where cv > 0
    group by 1
),
v2_cv_raw as (
    select
        campaign_id,
        slice(shuffle(array_agg(ad_id_plus)), 1, 100) as ad_ids
    from u2
    where cv > 0
    group by 1
),
v1_vimp_raw as (
    select
        campaign_id,
        slice(shuffle(array_agg(ad_id_plus)), 1, 1200) as ad_ids
    from u1
    where cv = 0
    group by 1
),
v2_vimp_raw as (
    select
        campaign_id,
        slice(shuffle(array_agg(ad_id_plus)), 1, 1200) as ad_ids
    from u2
    where cv = 0
    group by 1
),
v1_cv as (
    select
        *
    from v1_cv_raw
    where
        cardinality(ad_ids) >= 10
),
v2_cv as (
    select
        *
    from v2_cv_raw
    where
        cardinality(ad_ids) >= 10
),
v1_vimp as (
    select
        campaign_id,
        slice(a.ad_ids, 1, 12 * cardinality(b.ad_ids)) as ad_ids
    from v1_vimp_raw a
    inner join v1_cv_raw b using(campaign_id)
),
v2_vimp as (
    select
        campaign_id,
        slice(a.ad_ids, 1, 12 * cardinality(b.ad_ids)) as ad_ids
    from v2_vimp_raw a
    inner join v2_cv_raw b using(campaign_id)
)
select
    campaign_id,
    ad_id_plus,
    cast(0 as tinyint) as vimp1,
    cast(0 as tinyint) as vimp2,
    cast(1 as tinyint) as cv1,
    cast(0 as tinyint) as cv2
from v1_cv cross join unnest(ad_ids) as t(ad_id_plus)
union all
select
    campaign_id,
    ad_id_plus,
    cast(0 as tinyint) as vimp1,
    cast(0 as tinyint) as vimp2,
    cast(0 as tinyint) as cv1,
    cast(1 as tinyint) as cv2
from v2_cv cross join unnest(ad_ids) as t(ad_id_plus)
union all
select
    campaign_id,
    ad_id_plus,
    cast(1 as tinyint) as vimp1,
    cast(0 as tinyint) as vimp2,
    cast(0 as tinyint) as cv1,
    cast(0 as tinyint) as cv2
from v1_vimp cross join unnest(ad_ids) as t(ad_id_plus)
union all
select
    campaign_id,
    ad_id_plus,
    cast(0 as tinyint) as vimp1,
    cast(1 as tinyint) as vimp2,
    cast(0 as tinyint) as cv1,
    cast(0 as tinyint) as cv2
from v2_vimp cross join unnest(ad_ids) as t(ad_id_plus)
;
