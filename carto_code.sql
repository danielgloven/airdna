SELECT id, dt, country, ST_Shortestline(la_geom,review_geom) as line
FROM (
  SELECT a.cartodb_id id, a.review_date dt, a.country country, a.the_geom_webmercator la_geom,b.the_geom_webmercator review_geom FROM la_reviews_small_2 as a
  JOIN la_reviews_small as b
  ON a.cartodb_id=b.cartodb_id
  ) as links
