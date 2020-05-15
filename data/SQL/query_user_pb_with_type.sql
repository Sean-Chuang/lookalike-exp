delete from z_seanchuang.test_lal_offline_data;

insert into z_seanchuang.test_lal_offline_data
with u1 as (
    select
        c.click_action as type,
        ar.id as campaign_id,
        ar.ad_id as ad_id_plus,
        max(cast(ar.postback as tinyint)) as cv
    from ml.ad_result_v3 ar
    inner join rds_ad.smartad.campaign c
        on ar.id = c.campaign_id
    where
        ar.dt between '2020-04-15'
        and '2020-04-30'
        and (ar.opt_tg = 2 or ar.bid_t = 3)
    group by
        1,
        2,
        3
),
u2 as (
    select
        c.click_action as type,
        ar.id as campaign_id,
        ar.ad_id as ad_id_plus,
        max(cast(ar.postback as tinyint)) as cv
    from ml.ad_result_v3 ar
    inner join rds_ad.smartad.campaign c
        on ar.id = c.campaign_id
    where
        ar.dt between '2020-05-01'
        and '2020-05-14'
        and (ar.opt_tg = 2 or ar.bid_t = 3)
    group by
        1,
        2,
        3
),
v1_cv_raw as (
    select
        type,
        campaign_id,
        slice(shuffle(array_agg(ad_id_plus)), 1, 100) as ad_ids
    from u1
    where cv > 0
    group by 1, 2
),
v2_cv_raw as (
    select
        type,
        campaign_id,
        slice(shuffle(array_agg(ad_id_plus)), 1, 100) as ad_ids
    from u2
    where cv > 0
    group by 1, 2
),
v1_vimp_raw as (
    select
        type,
        campaign_id,
        slice(shuffle(array_agg(ad_id_plus)), 1, 1200) as ad_ids
    from u1
    where cv = 0
    group by 1, 2
),
v2_vimp_raw as (
    select
        type,
        campaign_id,
        slice(shuffle(array_agg(ad_id_plus)), 1, 1200) as ad_ids
    from u2
    where cv = 0
    group by 1, 2
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
        type,
        campaign_id,
        slice(a.ad_ids, 1, 12 * cardinality(b.ad_ids)) as ad_ids
    from v1_vimp_raw a
    inner join v1_cv_raw b using(type, campaign_id)
),
v2_vimp as (
    select
        type,
        campaign_id,
        slice(a.ad_ids, 1, 12 * cardinality(b.ad_ids)) as ad_ids
    from v2_vimp_raw a
    inner join v2_cv_raw b using(type, campaign_id)
)
select
    type,
    campaign_id,
    ad_id_plus,
    cast(0 as tinyint) as vimp1,
    cast(0 as tinyint) as vimp2,
    cast(1 as tinyint) as cv1,
    cast(0 as tinyint) as cv2
from v1_cv cross join unnest(ad_ids) as t(ad_id_plus)
union all
select
    type,
    campaign_id,
    ad_id_plus,
    cast(0 as tinyint) as vimp1,
    cast(0 as tinyint) as vimp2,
    cast(0 as tinyint) as cv1,
    cast(1 as tinyint) as cv2
from v2_cv cross join unnest(ad_ids) as t(ad_id_plus)
union all
select
    type,
    campaign_id,
    ad_id_plus,
    cast(1 as tinyint) as vimp1,
    cast(0 as tinyint) as vimp2,
    cast(0 as tinyint) as cv1,
    cast(0 as tinyint) as cv2
from v1_vimp cross join unnest(ad_ids) as t(ad_id_plus)
union all
select
    type,
    campaign_id,
    ad_id_plus,
    cast(0 as tinyint) as vimp1,
    cast(1 as tinyint) as vimp2,
    cast(0 as tinyint) as cv1,
    cast(0 as tinyint) as cv2
from v2_vimp cross join unnest(ad_ids) as t(ad_id_plus)
;
