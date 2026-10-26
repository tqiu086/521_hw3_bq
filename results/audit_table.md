| task / method | audit | N | % fail | acc if pass (n) | acc if fail (n) | Fisher p |
|---|---|---|---|---|---|---|
| gsm8k / crane | S1 input provenance | 200 | 15.0 | 82.4 (170) | 33.3 (30) | 1.5e-07 |
| gsm8k / crane | S2 step arithmetic recomputed | 200 | 38.0 | 83.9 (124) | 60.5 (76) | 0.00035 |
| gsm8k / crane | S3 answer = last step output | 200 | 23.5 | 88.9 (153) | 29.8 (47) | 1.2e-14 |
| gsm8k / template | S1 input provenance | 200 | 46.0 | 61.1 (108) | 56.5 (92) | 0.56 |
| gsm8k / template | S2 step arithmetic recomputed | 200 | 20.5 | 62.3 (159) | 46.3 (41) | 0.076 |
| gsm8k / template | S3 answer = last step output | 200 | 4.5 | 59.7 (191) | 44.4 (9) | 0.49 |
| bbh / crane | S1 input provenance | 200 | 0.0 | 36.0 (200) | nan (0) | nan |
| bbh / crane | S2 step arithmetic recomputed | 200 | 14.5 | 40.4 (171) | 10.3 (29) | 0.0014 |
| bbh / crane | S3 answer = last step output | 200 | 0.5 | 36.2 (199) | 0.0 (1) | 1 |
| bbh / template | S1 input provenance | 200 | 13.0 | 6.3 (174) | 3.8 (26) | 1 |
| bbh / template | S2 step arithmetic recomputed | 200 | 58.0 | 14.3 (84) | 0.0 (116) | 1.9e-05 |
| bbh / template | S3 answer = last step output | 200 | 1.0 | 6.1 (198) | 0.0 (2) | 1 |
| gsm8k / crane | T1 typicality (bottom 20% log-prob) | 170 | 20.0 | 80.9 (136) | 88.2 (34) | 0.45 |
| gsm8k / template | T1 typicality (bottom 20% log-prob) | 199 | 20.1 | 60.4 (159) | 55.0 (40) | 0.59 |
| bbh / crane | T1 typicality (bottom 20% log-prob) | 200 | 13.0 | 38.5 (174) | 19.2 (26) | 0.078 |
| bbh / template | T1 typicality (bottom 20% log-prob) | 200 | 20.0 | 5.6 (160) | 7.5 (40) | 0.71 |
| gsm8k / crane | I1 intervention (fail = answer unchanged) | 170 | 38.8 | 81.7 (104) | 83.3 (66) | 0.84 |
| bbh / crane | I1 intervention (fail = answer unchanged) | 200 | 6.0 | 35.1 (188) | 50.0 (12) | 0.36 |

Name clusters (k=6):
c0: 142 names, e.g. aday_apples, adults_smore, after_sister_steals, aisy_petals, amy_age, amy_friends, ankanis_and_licos_fish, ankanis_fish
c1: 48 names, e.g. add, after_use, compute, difference, first_parentheses, first_parenthesis, iced_subtotal, identity
c2: 59 names, e.g. adder, additional_cows, additional_time, adults_bars, anne_times, assign, big_coupons, caps
c3: 167 names, e.g. additional_crown_cost, additional_milk_needed, ads_total, amount_per_person, annual_bonus, annual_cost, annual_cost_miles, annual_miles
c4: 60 names, e.g. ads_year1, ads_year2, alex_weight, day1_infected, day2_infected, day3_infected, ellen_dog2_weight, ertotal_emails
c5: 101 names, e.g. amarin_apples_per_day, amber_badges_per_year, anancy_apples_per_day, arms_at_sides_taps_per_minute, arms_at_sides_total_taps, arms_raised_taps_per_minute, axies_gold_total, axies_silver_total
