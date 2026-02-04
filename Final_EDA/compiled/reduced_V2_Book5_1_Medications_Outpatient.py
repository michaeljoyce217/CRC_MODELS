# V2_Book5_1_Medications_Outpatient
# Functional cells: 27 of 64 code cells (123 total)
# Source: V2_Book5_1_Medications_Outpatient.ipynb
# =============================================================================

# ========================================
# CELL 1
# ========================================

# ---------------------------------
# Imports and Variable Declarations
# ---------------------------------

import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation

# Initialize a Spark session for distributed data processing
spark = SparkSession.builder.getOrCreate()

# Ensure date/time comparisons use Central Time
spark.conf.set("spark.sql.session.timeZone", "America/Chicago")

# Define target catalog for SQL based on the environment variable
trgt_cat = os.environ.get('trgt_cat')

# Use appropriate Spark catalog based on the target CATEGORY
spark.sql('USE CATALOG prod;')

# ========================================
# CELL 2
# ========================================

# Cell 1: Create medication grouper category map
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_med_grouper_category_map
USING DELTA
AS
WITH id_map AS (
  SELECT * FROM VALUES
    -- IRON SUPPLEMENTATION
    ('IRON_SUPPLEMENTATION', 103100859),
    ('IRON_SUPPLEMENTATION', 1060000647),
    ('IRON_SUPPLEMENTATION', 1060000646),
    ('IRON_SUPPLEMENTATION', 1060000650),

    -- PPI USE
    ('PPI_USE', 1030101319),
    ('PPI_USE', 103101245),
    ('PPI_USE', 1060033801),
    ('PPI_USE', 1060038401),

    -- NSAID/ASA USE
    ('NSAID_ASA_USE', 1060031001),
    ('NSAID_ASA_USE', 1060000523),
    ('NSAID_ASA_USE', 1060015801),
    ('NSAID_ASA_USE', 1060046101),
    ('NSAID_ASA_USE', 1060028401),
    ('NSAID_ASA_USE', 1060028501),
    ('NSAID_ASA_USE', 103100721),

    -- STATIN USE
    ('STATIN_USE', 1232000017),
    ('STATIN_USE', 1765734),
    ('STATIN_USE', 1765928),
    ('STATIN_USE', 1765229),
    ('STATIN_USE', 1765232),
    ('STATIN_USE', 1765246),
    ('STATIN_USE', 1765249),
    ('STATIN_USE', 1765260),
    ('STATIN_USE', 1765261),
    ('STATIN_USE', 1765262),
    ('STATIN_USE', 1765263),
    ('STATIN_USE', 1765264),
    ('STATIN_USE', 1754575),
    ('STATIN_USE', 1754577),
    ('STATIN_USE', 1754603),
    ('STATIN_USE', 1754584),
    ('STATIN_USE', 1754588),
    ('STATIN_USE', 1754592),
    ('STATIN_USE', 1754593),
    ('STATIN_USE', 1754594),
    ('STATIN_USE', 1754595),
    ('STATIN_USE', 1765241),
    ('STATIN_USE', 1765239),
    ('STATIN_USE', 1765240),
    ('STATIN_USE', 1765253),
    ('STATIN_USE', 1765254),
    ('STATIN_USE', 1030107485),
    ('STATIN_USE', 103103567),
    ('STATIN_USE', 103103088),
    ('STATIN_USE', 105100537),
    ('STATIN_USE', 1060048301),

    -- METFORMIN USE
    ('METFORMIN_USE', 103101190),
    ('METFORMIN_USE', 1060036201),
    ('METFORMIN_USE', 1060040001),
    ('METFORMIN_USE', 1060000704),
    ('METFORMIN_USE', 1060040101),
    ('METFORMIN_USE', 1765323),
    ('METFORMIN_USE', 1765276),
    ('METFORMIN_USE', 1765279),
    ('METFORMIN_USE', 1765282),
    ('METFORMIN_USE', 1765284),
    ('METFORMIN_USE', 1765289),
    ('METFORMIN_USE', 1765292),
    ('METFORMIN_USE', 1765320),
    ('METFORMIN_USE', 1765328),
    ('METFORMIN_USE', 1765331),
    ('METFORMIN_USE', 1765334),
    ('METFORMIN_USE', 1765336),
    ('METFORMIN_USE', 1765339)
  AS t(category_key, grouper_id)
)
SELECT
  m.category_key,
  gi.GROUPER_ID,
  gi.GROUPER_NAME
FROM id_map m
LEFT JOIN clarity.grouper_items gi
  ON gi.GROUPER_ID = m.grouper_id;
''')

# ========================================
# CELL 3
# ========================================

# Cell 2: Create medication ID category map (for medications not in groupers)
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_med_id_category_map
USING DELTA
AS
SELECT
  UPPER(TRIM(CATEGORY_KEY))   AS CATEGORY_KEY,
  CAST(MEDICATION_ID AS BIGINT) AS MEDICATION_ID,
  UPPER(TRIM(GEN_NAME))       AS GEN_NAME,
  SOURCE_TAG,
  NOTES
FROM VALUES
  -- ===== LAXATIVE_USE =====
  ('LAXATIVE_USE', 229101, 'PRUCALOPRIDE 2 MG TABLET',                  'regex_v1_vetted', '5-HT4 agonist'),
  ('LAXATIVE_USE', 229100, 'PRUCALOPRIDE 1 MG TABLET',                  'regex_v1_vetted', '5-HT4 agonist'),
  ('LAXATIVE_USE', 179778, 'PSYLLIUM HUSK 3.5 GRAM ORAL POWDER PACKET', 'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 221195, 'PSYLLIUM HUSK 0.4 GRAM CAPSULE',            'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 87600,  'PSYLLIUM HUSK 0.52 GRAM CAPSULE',           'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 81953,  'CALCIUM POLYCARBOPHIL 500 MG CHEWABLE TABLET','regex_v1_vetted','bulk/fiber'),
  ('LAXATIVE_USE', 80942,  'CALCIUM POLYCARBOPHIL 625 MG TABLET',       'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 80695,  'METHYLCELLULOSE (LAXATIVE) 500 MG TABLET',  'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 197631, 'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 39610,  'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 93851,  'LACTULOSE 10 GRAM/15 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 80411,  'LACTULOSE 10 GRAM ORAL PACKET',             'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 82328,  'LACTULOSE 20 GRAM ORAL PACKET',             'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 197575, 'LACTULOSE 20 GRAM/30 ML ORAL SOLUTION',     'regex_v1_vetted', 'osmotic'),
  ('LAXATIVE_USE', 95550,  'LUBIPROSTONE 24 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 178676, 'LUBIPROSTONE 8 MCG CAPSULE',                'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 70603,  'LUBIPROSTONE 24 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 214922, 'PSYLLIUM HUSK 3 GRAM/3 GRAM ORAL POWDER',   'regex_v1_vetted', 'bulk/fiber'),
  ('LAXATIVE_USE', 202759, 'LINACLOTIDE 145 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 202760, 'LINACLOTIDE 290 MCG CAPSULE',               'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 221253, 'LINACLOTIDE 72 MCG CAPSULE',                'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 221416, 'PLECANATIDE 3 MG TABLET',                   'regex_v1_vetted', 'secretagogue'),
  ('LAXATIVE_USE', 222935, 'NALDEMEDINE 0.2 MG TABLET',                 'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 212755, 'NALOXEGOL 12.5 MG TABLET',                  'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 212698, 'NALOXEGOL 25 MG TABLET',                    'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 219542, 'METHYLNALTREXONE 150 MG TABLET',            'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 178822, 'METHYLNALTREXONE 12 MG/0.6 ML SC SOLUTION', 'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 197993, 'METHYLNALTREXONE 12 MG/0.6 ML SC SYRINGE',  'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 197990, 'METHYLNALTREXONE 8 MG/0.4 ML SC SYRINGE',   'regex_v1_vetted', 'OIC antagonist'),
  ('LAXATIVE_USE', 2572,   'DOCUSATE SODIUM 100 MG TABLET',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 212170, 'DOCUSATE SODIUM 50 MG CAPSULE',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2085,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2568,   'DOCUSATE SODIUM 50 MG CAPSULE',             'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 2567,   'DOCUSATE SODIUM 250 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 1815,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 4372,   'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 11878,  'DOCUSATE SODIUM 100 MG CAPSULE',            'regex_v1_vetted', 'stool softener'),
  ('LAXATIVE_USE', 39752,  'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
  ('LAXATIVE_USE', 30015,  'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
  ('LAXATIVE_USE', 7159,   'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TAB','regex_v1_vetted','stimulant + stool softener'),
-- Polyethylene glycol products (very commonly used)
('LAXATIVE_USE', 25424, 'POLYETHYLENE GLYCOL 3350 17 GRAM ORAL POWDER PACKET', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 24984, 'POLYETHYLENE GLYCOL 3350 17 GRAM/DOSE ORAL POWDER', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 156129, 'POLYETHYLENE GLYCOL 3350 ORAL', 'discovery_v2', 'osmotic'),
-- Bisacodyl products
('LAXATIVE_USE', 1080, 'BISACODYL 10 MG RECTAL SUPPOSITORY', 'discovery_v2', 'stimulant'),
('LAXATIVE_USE', 13632, 'BISACODYL 5 MG TABLET,DELAYED RELEASE', 'discovery_v2', 'stimulant'),
('LAXATIVE_USE', 83533, 'BISACODYL 5 MG TABLET', 'discovery_v2', 'stimulant'),
-- Magnesium products
('LAXATIVE_USE', 79944, 'MAGNESIUM HYDROXIDE 400 MG/5 ML ORAL SUSPENSION', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 4711, 'MAGNESIUM CITRATE ORAL SOLUTION', 'discovery_v2', 'osmotic'),
('LAXATIVE_USE', 155118, 'MAGNESIUM CITRATE ORAL', 'discovery_v2', 'osmotic'),
-- Senna combinations (beyond what you have)
('LAXATIVE_USE', 40926, 'SENNOSIDES 8.6 MG-DOCUSATE SODIUM 50 MG TABLET', 'discovery_v2', 'stimulant combo'),
-- Alvimopan (opioid antagonist for postop ileus)
('LAXATIVE_USE', 179661, 'ALVIMOPAN 12 MG CAPSULE', 'discovery_v2', 'peripherally acting opioid antagonist'),
-- Add these to ANTIDIARRHEAL_USE
('ANTIDIARRHEAL_USE', 4560, 'LOPERAMIDE 2 MG CAPSULE', 'discovery_v2', 'antimotility'),
('ANTIDIARRHEAL_USE', 4562, 'LOPERAMIDE 2 MG TABLET', 'discovery_v2', 'antimotility'),
('ANTIDIARRHEAL_USE', 189801, 'RIFAXIMIN 550 MG TABLET', 'discovery_v2', 'antibiotic for IBS-D/HE'),
('ANTIDIARRHEAL_USE', 189817, 'RIFAXIMIN 550 MG TABLET (XIFAXAN)', 'discovery_v2', 'antibiotic for IBS-D/HE'),
('ANTIDIARRHEAL_USE', 81342, 'BISMUTH SUBSALICYLATE 262 MG CHEWABLE TABLET', 'discovery_v2', 'bismuth compound'),
('ANTIDIARRHEAL_USE', 80560, 'BISMUTH SUBSALICYLATE 262 MG/15 ML ORAL SUSPENSION', 'discovery_v2', 'bismuth compound'),
('ANTIDIARRHEAL_USE', 89923, 'ALOSETRON 0.5 MG TABLET', 'discovery_v2', '5-HT3 antagonist for IBS-D'),
('ANTIDIARRHEAL_USE', 79963, 'ALOSETRON 1 MG TABLET', 'discovery_v2', '5-HT3 antagonist for IBS-D'),

  -- ===== ANTIDIARRHEAL_USE =====
  ('ANTIDIARRHEAL_USE', 2516,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG TABLET',             'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 88489,  'CHOLESTYRAMINE-ASPARTAME 4 GRAM ORAL POWDER FOR SUSP IN A PACKET', 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 183910, 'COLESEVELAM 3.75 GRAM ORAL POWDER PACKET',                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 16388,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 37482,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 80288,  'COLESEVELAM 625 MG TABLET',                                 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176079, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 82389,  'COLESTIPOL 5 GRAM ORAL PACKET',                             'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 21994,  'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216207, 'ELUXADOLINE 75 MG TABLET',                                  'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 9589,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM ORAL POWDER',            'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 89845,  'COLESTIPOL 5 GRAM ORAL GRANULES',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176078, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 28857,  'COLESEVELAM 625 MG TABLET',                                 'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216198, 'ELUXADOLINE 100 MG TABLET',                                 'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 82555,  'COLESTIPOL 1 GRAM TABLET',                                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 9588,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM POWDER FOR SUSP IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 176054, 'CHOLESTYRAMINE-ASPARTAME 4 GRAM ORAL POWDER',               'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 13898,  'COLESTIPOL 1 GRAM TABLET',                                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216086, 'ELUXADOLINE 75 MG TABLET',                                  'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 2515,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG/5 ML ORAL LIQUID',   'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 216087, 'ELUXADOLINE 100 MG TABLET',                                 'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 183915, 'COLESEVELAM 3.75 GRAM ORAL POWDER PACKET',                  'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 252231, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER',                         'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 252230, 'CHOLESTYRAMINE 4 GRAM ORAL POWDER FOR SUSPENSION IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 216374, 'ELUXADOLINE ORAL',                                          'regex_v1_vetted', 'IBS-D agent'),
  ('ANTIDIARRHEAL_USE', 4553,   'DIPHENOXYLATE-ATROPINE 2.5 MG-0.025 MG TABLET',             'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 154828, 'DIPHENOXYLATE-ATROPINE ORAL',                               'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 146632, 'CHOLESTYRAMINE-ASPARTAME ORAL',                             'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147147, 'COLESTIPOL ORAL',                                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 6767,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM ORAL POWDER',            'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 160917, 'CHOLESTYRAMINE (WITH SUGAR) ORAL',                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 166274, 'COLESEVELAM ORAL',                                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 89844,  'COLESTIPOL 7.5 GRAM ORAL PACKET',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147146, 'COLESTIPOL ORAL',                                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 148928, 'DIPHENOXYLATE-ATROPINE ORAL',                               'regex_v1_vetted', 'vetted antidiarrheal'),
  ('ANTIDIARRHEAL_USE', 12057,  'COLESTIPOL 7.5 GRAM ORAL PACKET',                           'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 197067, 'CHOLESTYRAMINE (WITH SUGAR) ORAL',                          'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 146631, 'CHOLESTYRAMINE ORAL',                                       'regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 6766,   'CHOLESTYRAMINE (WITH SUGAR) 4 GRAM POWDER FOR SUSP IN A PACKET','regex_v1_vetted', 'bile-acid binder'),
  ('ANTIDIARRHEAL_USE', 147144, 'COLESEVELAM ORAL',                                          'regex_v1_vetted', 'bile-acid binder'),

  -- ===== ANTISPASMODIC_USE =====
  ('ANTISPASMODIC_USE', 1738,    'CHLORDIAZEPOXIDE-CLIDINIUM 5 MG-2.5 MG CAPSULE', 'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 2418,    'DICYCLOMINE 10 MG CAPSULE',                      'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 205840,  'DICYCLOMINE 10 MG/5 ML ORAL SOLUTION',           'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 2420,    'DICYCLOMINE 20 MG TABLET',                       'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 144535,  'DICYCLOMINE ORAL',                               'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 148732,  'DICYCLOMINE ORAL',                               'regex_v1_vetted', 'GI anticholinergic'),
  ('ANTISPASMODIC_USE', 4434,    'CHLORDIAZEPOXIDE-CLIDINIUM 5 MG-2.5 MG CAPSULE', 'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 146851,  'CHLORDIAZEPOXIDE-CLIDINIUM ORAL',                'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 154497,  'CHLORDIAZEPOXIDE-CLIDINIUM ORAL',                'regex_v1_vetted', 'Librax combo'),
  ('ANTISPASMODIC_USE', 254015,  'DICYCLOMINE 40 MG TABLET',                       'regex_v1_vetted', 'GI anticholinergic'),

  -- ===== B12_FOLATE_USE =====
  ('B12_FOLATE_USE', 4396,   'LEUCOVORIN CALCIUM 15 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 4397,   'LEUCOVORIN CALCIUM 25 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 212837, 'LEUCOVORIN 4 MG-PYRIDOXAL PHOSPHATE 50 MG-MECOBALAMIN 2 MG TABLET','regex_v1_vetted','combo w/ folinic acid & mecobalamin'),
  ('B12_FOLATE_USE', 4398,   'LEUCOVORIN CALCIUM 5 MG TABLET',                       'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 4395,   'LEUCOVORIN CALCIUM 10 MG TABLET',                      'regex_v1_vetted', 'folinic acid tablet'),
  ('B12_FOLATE_USE', 242499, 'METHYLFOLATE CALCIUM 25,000 MCG DFE-MECOBALAMIN 2,000 MCG CAPSULE','regex_v1_vetted','high-dose methylfolate + mecobalamin'),
  ('B12_FOLATE_USE', 243320, 'B12 5,000 MCG-METHYLFOLATE 1,360 MCG DFE-B6 2.5 MG CHEWABLE TABLET','regex_v1_vetted','high-dose B12 + methylfolate combo'),
  ('B12_FOLATE_USE', 154414, 'LEUCOVORIN CALCIUM ORAL',                             'regex_v1_vetted', 'folinic acid oral'),
  ('B12_FOLATE_USE', 15370,  'LEUCOVORIN CALCIUM 10 MG/ML INJECTION SOLUTION',      'regex_v1_vetted', 'folinic acid injection'),
  ('B12_FOLATE_USE', 216354, 'COQ10-ALA-RESVERATROL-LEUCOVORIN-B6-MECOBALAMIN-VITC-D3 ORAL','regex_v1_vetted','multi-ingredient incl. leucovorin & mecobalamin'),
  ('B12_FOLATE_USE', 4394,   'LEUCOVORIN CALCIUM 50 MG SOLUTION FOR INJECTION',     'regex_v1_vetted', 'folinic acid injection'),
  ('B12_FOLATE_USE', 4392,   'LEUCOVORIN CALCIUM 100 MG SOLUTION FOR INJECTION',    'regex_v1_vetted', 'folinic acid injection'),

-- ===== IBD_MED_USE =====
('IBD_MEDICATIONS', 6496, 'PREDNISONE 20 MG TABLET', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 6494, 'PREDNISONE 10 MG TABLET', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 14887, 'METHYLPREDNISOLONE 4 MG TABLETS IN A DOSE PACK', 'discovery_v2', 'corticosteroid'),
('IBD_MEDICATIONS', 82000, 'BUDESONIDE DR - ER 3 MG CAPSULE,DELAYED,EXTENDED RELEASE', 'discovery_v2', 'GI-specific steroid'),
('IBD_MEDICATIONS', 167329, 'MESALAMINE 1.2 GRAM TABLET,DELAYED RELEASE', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 94087, 'MESALAMINE 1,000 MG RECTAL SUPPOSITORY', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 10535, 'MESALAMINE 4 GRAM/60 ML ENEMA', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 7562, 'SULFASALAZINE 500 MG TABLET', 'discovery_v2', '5-ASA'),
('IBD_MEDICATIONS', 9183, 'AZATHIOPRINE 50 MG TABLET', 'discovery_v2', 'immunosuppressant'),
('IBD_MEDICATIONS', 4973, 'METHOTREXATE SODIUM 2.5 MG TABLET', 'discovery_v2', 'immunosuppressant'),
('IBD_MEDICATIONS', 80729, 'INFLIXIMAB 100 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'anti-TNF biologic'),
('IBD_MEDICATIONS', 225598, 'ADALIMUMAB 40 MG/0.4 ML SUBCUTANEOUS PEN KIT', 'discovery_v2', 'anti-TNF biologic'),
('IBD_MEDICATIONS', 209779, 'VEDOLIZUMAB 300 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'α4β7 integrin blocker'),
('IBD_MEDICATIONS', 184073, 'USTEKINUMAB 90 MG/ML SUBCUTANEOUS SYRINGE', 'discovery_v2', 'IL-12/23 inhibitor'),
('IBD_MEDICATIONS', 217804, 'TOFACITINIB ER 11 MG TABLET,EXTENDED RELEASE 24 HR', 'discovery_v2', 'JAK inhibitor'),

-- ===== HEMORRHOID/RECTAL_MED_USE =====
('HEMORRHOID_RECTAL_MEDS', 3738, 'HYDROCORTISONE ACETATE 25 MG RECTAL SUPPOSITORY', 'discovery_v2', 'rectal steroid'),
('HEMORRHOID_RECTAL_MEDS', 6589, 'HYDROCORTISONE 1 %-PRAMOXINE 1 % RECTAL FOAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 77665, 'HYDROCORTISONE-PRAMOXINE 2.5 %-1 % RECTAL CREAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 80624, 'HYDROCORTISONE 1 %-PRAMOXINE 1 % RECTAL FOAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 82892, 'HYDROCORTISONE-PRAMOXINE 1 %-1 % RECTAL CREAM', 'discovery_v2', 'rectal steroid combo'),
('HEMORRHOID_RECTAL_MEDS', 10210, 'HYDROCORTISONE 100 MG/60 ML ENEMA', 'discovery_v2', 'rectal steroid'),
('HEMORRHOID_RECTAL_MEDS', 35164, 'LIDOCAINE 3 %-HYDROCORTISONE 0.5 % RECTAL CREAM', 'discovery_v2', 'anesthetic combo'),

-- ===== GI_BLEEDING_MED_USE =====
('GI_BLEEDING_MEDS', 230527, 'TRANEXAMIC ACID 1,000 MG/100 ML IV PIGGYBACK', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 196970, 'TRANEXAMIC ACID 1,000 MG/10 ML INTRAVENOUS SOLUTION', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 190130, 'TRANEXAMIC ACID 650 MG TABLET', 'discovery_v2', 'antifibrinolytic'),
('GI_BLEEDING_MEDS', 211947, 'VASOPRESSIN 20 UNIT/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'vasoconstrictor'),
('GI_BLEEDING_MEDS', 25122, 'OCTREOTIDE ACETATE 100 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'somatostatin analog'),
('GI_BLEEDING_MEDS', 25121, 'OCTREOTIDE ACETATE 50 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'somatostatin analog'),

-- ===== CHRONIC_OPIOD_MED_USE =====
('CHRONIC_OPIOID_USE', 3037, 'FENTANYL 50 MCG/ML INJECTION SOLUTION', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 34505, 'HYDROCODONE 5 MG-ACETAMINOPHEN 325 MG TABLET', 'discovery_v2', 'opioid combo'),
('CHRONIC_OPIOID_USE', 10814, 'OXYCODONE 5 MG TABLET', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 301537, 'MORPHINE 4 MG/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 14632, 'TRAMADOL 50 MG TABLET', 'discovery_v2', 'opioid'),
('CHRONIC_OPIOID_USE', 27905, 'FENTANYL 25 MCG/HR TRANSDERMAL PATCH', 'discovery_v2', 'long-acting opioid'),
('CHRONIC_OPIOID_USE', 20920, 'MORPHINE ER 15 MG TABLET,EXTENDED RELEASE', 'discovery_v2', 'long-acting opioid'),
('CHRONIC_OPIOID_USE', 211843, 'OXYCODONE ER 10 MG TABLET,CRUSH RESISTANT,EXTENDED RELEASE', 'discovery_v2', 'long-acting opioid'),

-- ===== BROAD_SPECTRUM_ANTIBIOTIC_USE =====
('BROAD_SPECTRUM_ANTIBIOTICS', 9500, 'CEPHALEXIN 500 MG CAPSULE', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 25119, 'CIPROFLOXACIN 500 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 82091, 'LEVOFLOXACIN 500 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 87765, 'MOXIFLOXACIN 400 MG TABLET', 'discovery_v2', 'fluoroquinolone'),
('BROAD_SPECTRUM_ANTIBIOTICS', 177634, 'CEFAZOLIN 2 GRAM/20 ML INTRAVENOUS SYRINGE', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 79742, 'CEFTRIAXONE 1 GRAM SOLUTION FOR INJECTION', 'discovery_v2', 'cephalosporin'),
('BROAD_SPECTRUM_ANTIBIOTICS', 9621, 'CLINDAMYCIN HCL 300 MG CAPSULE', 'discovery_v2', 'lincosamide'),
('BROAD_SPECTRUM_ANTIBIOTICS', 83455, 'PIPERACILLIN-TAZOBACTAM 3.375 GRAM IV PIGGYBACK', 'discovery_v2', 'beta-lactam combo'),
('BROAD_SPECTRUM_ANTIBIOTICS', 83077, 'ERTAPENEM 1 GRAM SOLUTION FOR INJECTION', 'discovery_v2', 'carbapenem'),
('BROAD_SPECTRUM_ANTIBIOTICS', 80713, 'MEROPENEM 1 GRAM INTRAVENOUS SOLUTION', 'discovery_v2', 'carbapenem'),

-- ===== HORMON_THERAPY_USE =====
('HORMONE_THERAPY', 82101, 'ESTRADIOL 0.01% VAGINAL CREAM', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 9967, 'ESTRADIOL 1 MG TABLET', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 80522, 'CONJUGATED ESTROGENS 0.625 MG/GRAM VAGINAL CREAM', 'discovery_v2', 'estrogen'),
('HORMONE_THERAPY', 11498, 'TAMOXIFEN 20 MG TABLET', 'discovery_v2', 'SERM'),
('HORMONE_THERAPY', 81433, 'RALOXIFENE 60 MG TABLET', 'discovery_v2', 'SERM'),
('HORMONE_THERAPY', 7784, 'TESTOSTERONE CYPIONATE 200 MG/ML INTRAMUSCULAR OIL', 'discovery_v2', 'testosterone'),
('HORMONE_THERAPY', 194705, 'TESTOSTERONE 1.62% TRANSDERMAL GEL', 'discovery_v2', 'testosterone'),

-- ===== CHEMOTHERAPY_AGENT_USE =====
('CHEMOTHERAPY_AGENTS', 78342, 'FLUOROURACIL 5 % TOPICAL CREAM', 'discovery_v2', '5-FU'),
('CHEMOTHERAPY_AGENTS', 79057, 'CAPECITABINE 500 MG TABLET', 'discovery_v2', '5-FU prodrug'),
('CHEMOTHERAPY_AGENTS', 80994, 'CAPECITABINE 150 MG TABLET', 'discovery_v2', '5-FU prodrug'),
('CHEMOTHERAPY_AGENTS', 89931, 'BEVACIZUMAB 25 MG/ML INTRAVENOUS SOLUTION', 'discovery_v2', 'VEGF inhibitor'),
('CHEMOTHERAPY_AGENTS', 10631, 'MITOMYCIN 40 MG INTRAVENOUS SOLUTION', 'discovery_v2', 'alkylating agent')

AS t(category_key, medication_id, gen_name, source_tag, notes);
''')

# ========================================
# CELL 4
# ========================================

# Cell 3: Create unpivoted outpatient medications table
# This cell extracts outpatient medication orders from the EHR and categorizes them
# based on our expanded medication mappings for CRC risk prediction
# CRITICAL: Only includes orders from July 1, 2021 forward (data availability constraint)

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds_unpivoted AS

WITH
cohort AS (
  -- Get all patient-month observations from our cohort
  SELECT
    CAST(PAT_ID AS STRING)          AS PAT_ID,
    END_DTTM
  FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
),

map_med AS (
  -- Load direct medication ID to category mappings
  -- These are medications identified through discovery queries
  SELECT
    UPPER(TRIM(CATEGORY_KEY))       AS CATEGORY,
    CAST(MEDICATION_ID AS BIGINT)   AS MEDICATION_ID,
    UPPER(TRIM(GEN_NAME))           AS GEN_NAME
  FROM {trgt_cat}.clncl_ds.herald_eda_train_med_id_category_map
),

map_grp AS (
  -- Load grouper-based medication category mappings
  -- Groupers are pre-defined medication sets maintained by the institution
  SELECT
    UPPER(TRIM(CATEGORY_KEY))       AS CATEGORY,
    CAST(GROUPER_ID AS BIGINT)      AS GROUPER_ID,
    UPPER(TRIM(GROUPER_NAME))       AS GROUPER_NAME
  FROM {trgt_cat}.clncl_ds.herald_eda_train_med_grouper_category_map
),

-- Compiled membership: expand groupers to individual medications
-- This creates the GROUPER_ID → MEDICATION_ID mapping
grp_med_members AS (
  SELECT DISTINCT
    CAST(itm.GROUPER_ID AS BIGINT)                AS GROUPER_ID,
    CAST(med.MEDICATION_ID AS BIGINT)             AS MEDICATION_ID
  FROM map_grp mg
  JOIN clarity.grouper_items itm
    ON itm.GROUPER_ID = mg.GROUPER_ID
  JOIN clarity.grouper_compiled_rec_list rec
    ON rec.base_grouper_id = itm.GROUPER_ID
  JOIN clarity.clarity_medication med
    ON med.MEDICATION_ID = rec.grouper_records_numeric_id
),

-- Extract outpatient medication orders
-- Key differences from inpatient: uses ORDER data not MAR (medication administration)
-- CRITICAL: Enforces July 1, 2021 data availability constraint
orders_outpatient AS (
  SELECT
    CAST(ome.PAT_ID AS STRING)                    AS PAT_ID,
    CAST(ome.MEDICATION_ID AS BIGINT)             AS MEDICATION_ID,
    UPPER(TRIM(ome.GENERIC_NAME))                 AS RAW_GENERIC,
    CAST(ome.ORDER_START_TIME AS TIMESTAMP)       AS ORDER_TIME,
    CAST(ome.ORDER_MED_ID AS BIGINT)              AS ORDER_MED_ID
  FROM clarity_cur.order_med_enh ome
  WHERE ome.ORDERING_MODE_C <> 2                  -- Exclude inpatient orders
    AND ome.ORDER_START_TIME IS NOT NULL
    AND DATE(ome.ORDER_START_TIME) >= '2021-07-01'  -- Data availability constraint from Book0
    AND ome.ORDER_CLASS <> 'Historical Med'       -- Exclude historical reconciliation
    AND ome.ORDER_STATUS_C IN (2, 5)              -- Sent (2) or Completed (5) orders only
),

-- Match medications via direct medication ID mapping
hits_med AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    mm.CATEGORY,
    om.MEDICATION_ID,
    om.RAW_GENERIC,
    om.ORDER_TIME
  FROM cohort c
  LEFT JOIN orders_outpatient om
    ON om.PAT_ID = c.PAT_ID
   AND DATE(om.ORDER_TIME) <  c.END_DTTM          -- Before prediction point
   AND DATE(om.ORDER_TIME) >= ADD_MONTHS(c.END_DTTM, -24)  -- Within 24-month lookback
  JOIN map_med mm
    ON mm.MEDICATION_ID = om.MEDICATION_ID
),

-- Match medications via grouper membership
hits_grp AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    mg.CATEGORY,
    om.MEDICATION_ID,
    om.RAW_GENERIC,
    om.ORDER_TIME
  FROM cohort c
  LEFT JOIN orders_outpatient om
    ON om.PAT_ID = c.PAT_ID
   AND DATE(om.ORDER_TIME) <  c.END_DTTM
   AND DATE(om.ORDER_TIME) >= ADD_MONTHS(c.END_DTTM, -24)
  JOIN grp_med_members gm
    ON gm.MEDICATION_ID = om.MEDICATION_ID
  JOIN map_grp mg
    ON mg.GROUPER_ID = gm.GROUPER_ID
),

-- Combine all medication category matches
hits_all AS (
  SELECT * FROM hits_med
  UNION ALL
  SELECT * FROM hits_grp
),

-- Deduplicate: Keep most recent order per medication per day
-- This handles cases where same medication ordered multiple times in a day
ranked AS (
  SELECT
    PAT_ID,
    END_DTTM,
    CATEGORY,
    MEDICATION_ID,
    RAW_GENERIC,
    ORDER_TIME,
    DATEDIFF(END_DTTM, DATE(ORDER_TIME)) AS DAYS_SINCE_MED,  -- Days from order to prediction
    ROW_NUMBER() OVER (
      PARTITION BY PAT_ID, END_DTTM, CATEGORY, MEDICATION_ID, DATE(ORDER_TIME)
      ORDER BY ORDER_TIME DESC
    ) AS rn
  FROM hits_all
)

-- Final output: one row per patient-observation-category-medication-day
SELECT
  c.PAT_ID,
  c.END_DTTM,
  r.CATEGORY,
  r.MEDICATION_ID,
  r.RAW_GENERIC,
  r.ORDER_TIME,
  r.DAYS_SINCE_MED
FROM cohort c
LEFT JOIN ranked r
  ON r.PAT_ID   = c.PAT_ID
 AND r.END_DTTM = c.END_DTTM
WHERE r.rn = 1 OR r.rn IS NULL;
''')

print("Outpatient medications unpivoted table created with July 2021 data constraint")

# ========================================
# CELL 5
# ========================================

# Grouper Expansion Validation

# Verify that grouper-based medication categories are properly expanding to individual medications

# Cell 3A: Validate grouper expansion logic
print("="*70)
print("GROUPER EXPANSION VALIDATION")
print("="*70)

# Check how many medications each grouper category expands to
grouper_expansion = spark.sql(f"""
WITH grp_med_members AS (
  SELECT DISTINCT
    CAST(itm.GROUPER_ID AS BIGINT) AS GROUPER_ID,
    CAST(med.MEDICATION_ID AS BIGINT) AS MEDICATION_ID
  FROM {trgt_cat}.clncl_ds.herald_eda_train_med_grouper_category_map mg
  JOIN clarity.grouper_items itm
    ON itm.GROUPER_ID = mg.GROUPER_ID
  JOIN clarity.grouper_compiled_rec_list rec
    ON rec.base_grouper_id = itm.GROUPER_ID
  JOIN clarity.clarity_medication med
    ON med.MEDICATION_ID = rec.grouper_records_numeric_id
)
SELECT 
  mg.CATEGORY_KEY,
  mg.GROUPER_NAME,
  COUNT(DISTINCT gm.MEDICATION_ID) as n_medications_expanded
FROM {trgt_cat}.clncl_ds.herald_eda_train_med_grouper_category_map mg
LEFT JOIN grp_med_members gm 
  ON mg.GROUPER_ID = gm.GROUPER_ID
GROUP BY mg.CATEGORY_KEY, mg.GROUPER_NAME
ORDER BY n_medications_expanded DESC
""")

display(grouper_expansion)

# Validate that groupers actually returned medications
grouper_check = grouper_expansion.collect()
for row in grouper_check:
    if row['n_medications_expanded'] == 0:
        print(f"WARNING: Grouper '{row['GROUPER_NAME']}' expanded to 0 medications!")
    else:
        print(f"✓ {row['CATEGORY_KEY']}: {row['n_medications_expanded']} medications from grouper")

print("="*70)

# ========================================
# CELL 6
# ========================================

# Cell 3B: Validate no temporal leakage and proper date filtering
print("="*70)
print("TEMPORAL INTEGRITY VALIDATION")
print("="*70)

# Check 1: Verify no future leakage (ORDER_TIME should always be < END_DTTM)
leakage_check = spark.sql(f"""
SELECT COUNT(*) as future_leakage_violations
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds_unpivoted
WHERE ORDER_TIME >= END_DTTM
""").collect()[0]

print(f"\n1. Future Leakage Check:")
print(f"   Violations (ORDER_TIME >= END_DTTM): {leakage_check['future_leakage_violations']:,}")
if leakage_check['future_leakage_violations'] == 0:
    print("   ✓ PASS: No temporal leakage detected")
else:
    print("   ✗ FAIL: Temporal leakage detected!")

# Check 2: Verify 24-month lookback window
window_check = spark.sql(f"""
SELECT 
  MIN(DAYS_SINCE_MED) as min_days_back,
  MAX(DAYS_SINCE_MED) as max_days_back,
  AVG(DAYS_SINCE_MED) as avg_days_back
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds_unpivoted
WHERE ORDER_TIME IS NOT NULL
""").collect()[0]

print(f"\n2. Lookback Window Validation:")
print(f"   Min days back: {window_check['min_days_back']}")
print(f"   Max days back: {window_check['max_days_back']}")
print(f"   Avg days back: {window_check['avg_days_back']:.1f}")
expected_max = 24 * 30 + 30  # ~750 days (24 months + buffer)
if window_check['max_days_back'] <= expected_max:
    print(f"   ✓ PASS: Max within 24-month window (≤{expected_max} days)")
else:
    print(f"   ✗ FAIL: Medications beyond 24-month window detected!")

# Check 3: Verify July 2021 data constraint
date_constraint_check = spark.sql(f"""
SELECT 
  MIN(DATE(ORDER_TIME)) as earliest_order_date,
  COUNT(*) as total_medication_orders,
  SUM(CASE WHEN DATE(ORDER_TIME) < '2021-07-01' THEN 1 ELSE 0 END) as pre_july_2021_orders
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds_unpivoted
WHERE ORDER_TIME IS NOT NULL
""").collect()[0]

print(f"\n3. July 2021 Data Constraint Check:")
print(f"   Earliest order date: {date_constraint_check['earliest_order_date']}")
print(f"   Pre-July 2021 orders: {date_constraint_check['pre_july_2021_orders']:,}")
if date_constraint_check['pre_july_2021_orders'] == 0:
    print("   ✓ PASS: All orders from July 1, 2021 or later")
else:
    print(f"   ✗ FAIL: {date_constraint_check['pre_july_2021_orders']:,} orders before July 2021!")

print("="*70)

# ========================================
# CELL 7
# ========================================

# Cell 4: Create final pivoted outpatient medications table using window functions
# This cell transforms the unpivoted medication data into features for modeling
# Creates 3 types of features per category: flag (ever used), days_since (recency), count (frequency)

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds AS

WITH
  cohort AS (
    -- Base cohort with all patient-month observations
    SELECT CAST(PAT_ID AS STRING) AS PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
  ),

  unpvt AS (
    -- Clean and filter the unpivoted medication data
    SELECT CAST(PAT_ID AS STRING) AS PAT_ID,
           END_DTTM,
           UPPER(TRIM(CATEGORY)) AS CATEGORY,
           CAST(DAYS_SINCE_MED AS INT) AS DAYS_SINCE_MED
    FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds_unpivoted
    WHERE CATEGORY IS NOT NULL
  ),

  -- RECENCY FEATURES: Find most recent medication use per category
  ranked_meds AS (
    SELECT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      DAYS_SINCE_MED,
      ROW_NUMBER() OVER (
        PARTITION BY PAT_ID, END_DTTM, CATEGORY 
        ORDER BY DAYS_SINCE_MED ASC  -- Smallest value = most recent
      ) AS rn_most_recent
    FROM unpvt
  ),

  most_recent_meds AS (
    SELECT 
      PAT_ID,
      END_DTTM,
      CATEGORY,
      DAYS_SINCE_MED
    FROM ranked_meds
    WHERE rn_most_recent = 1
  ),

  -- FREQUENCY FEATURES: Count total medication orders per category
  med_counts AS (
    SELECT DISTINCT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      COUNT(*) OVER (
        PARTITION BY PAT_ID, END_DTTM, CATEGORY
      ) AS med_count
    FROM unpvt
  ),

  -- BINARY FLAGS: Has patient ever used this medication category?
  med_flags AS (
    SELECT DISTINCT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      1 AS has_med_flag
    FROM unpvt
  ),

  -- Pivot recency features (days since last use)
  pivot_days AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Existing categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN DAYS_SINCE_MED END) AS iron_use_days_since,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN DAYS_SINCE_MED END) AS b12_or_folate_use_days_since,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN DAYS_SINCE_MED END) AS laxative_use_days_since,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN DAYS_SINCE_MED END) AS antidiarrheal_use_days_since,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN DAYS_SINCE_MED END) AS antispasmodic_use_days_since,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN DAYS_SINCE_MED END) AS ppi_use_days_since,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN DAYS_SINCE_MED END) AS nsaid_asa_use_days_since,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN DAYS_SINCE_MED END) AS statin_use_days_since,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN DAYS_SINCE_MED END) AS metformin_use_days_since,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN DAYS_SINCE_MED END) AS ibd_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN DAYS_SINCE_MED END) AS hemorrhoid_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN DAYS_SINCE_MED END) AS gi_bleed_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN DAYS_SINCE_MED END) AS opioid_use_days_since,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN DAYS_SINCE_MED END) AS broad_abx_days_since,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN DAYS_SINCE_MED END) AS hormone_therapy_days_since,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN DAYS_SINCE_MED END) AS chemo_agents_days_since
    FROM most_recent_meds
    GROUP BY PAT_ID, END_DTTM
  ),

  -- Pivot frequency features (count of orders in 2 years)
  pivot_counts AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Existing categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN med_count END) AS iron_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN med_count END) AS b12_or_folate_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN med_count END) AS laxative_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN med_count END) AS antidiarrheal_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN med_count END) AS antispasmodic_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN med_count END) AS ppi_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN med_count END) AS nsaid_asa_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN med_count END) AS statin_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN med_count END) AS metformin_use_count_2yr,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN med_count END) AS ibd_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN med_count END) AS hemorrhoid_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN med_count END) AS gi_bleed_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN med_count END) AS opioid_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN med_count END) AS broad_abx_count_2yr,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN med_count END) AS hormone_therapy_count_2yr,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN med_count END) AS chemo_agents_count_2yr
    FROM med_counts
    GROUP BY PAT_ID, END_DTTM
  ),

  -- Pivot binary flags (ever used in 2 years)
  pivot_flags AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Existing categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN has_med_flag END) AS iron_use_flag,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN has_med_flag END) AS b12_or_folate_use_flag,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN has_med_flag END) AS laxative_use_flag,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN has_med_flag END) AS antidiarrheal_use_flag,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN has_med_flag END) AS antispasmodic_use_flag,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN has_med_flag END) AS ppi_use_flag,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN has_med_flag END) AS nsaid_asa_use_flag,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN has_med_flag END) AS statin_use_flag,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN has_med_flag END) AS metformin_use_flag,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN has_med_flag END) AS ibd_meds_flag,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN has_med_flag END) AS hemorrhoid_meds_flag,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN has_med_flag END) AS gi_bleed_meds_flag,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN has_med_flag END) AS opioid_use_flag,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN has_med_flag END) AS broad_abx_flag,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN has_med_flag END) AS hormone_therapy_flag,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN has_med_flag END) AS chemo_agents_flag
    FROM med_flags
    GROUP BY PAT_ID, END_DTTM
  )

-- Final assembly: Join all feature types
-- COALESCE ensures 0 for flags/counts when no medication found (instead of NULL)
SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- Existing medication features
  COALESCE(pf.iron_use_flag, 0) AS iron_use_flag,
  pd.iron_use_days_since,
  COALESCE(pc.iron_use_count_2yr, 0) AS iron_use_count_2yr,

  COALESCE(pf.b12_or_folate_use_flag, 0) AS b12_or_folate_use_flag,
  pd.b12_or_folate_use_days_since,
  COALESCE(pc.b12_or_folate_use_count_2yr, 0) AS b12_or_folate_use_count_2yr,

  COALESCE(pf.laxative_use_flag, 0) AS laxative_use_flag,
  pd.laxative_use_days_since,
  COALESCE(pc.laxative_use_count_2yr, 0) AS laxative_use_count_2yr,

  COALESCE(pf.antidiarrheal_use_flag, 0) AS antidiarrheal_use_flag,
  pd.antidiarrheal_use_days_since,
  COALESCE(pc.antidiarrheal_use_count_2yr, 0) AS antidiarrheal_use_count_2yr,

  COALESCE(pf.antispasmodic_use_flag, 0) AS antispasmodic_use_flag,
  pd.antispasmodic_use_days_since,
  COALESCE(pc.antispasmodic_use_count_2yr, 0) AS antispasmodic_use_count_2yr,

  COALESCE(pf.ppi_use_flag, 0) AS ppi_use_flag,
  pd.ppi_use_days_since,
  COALESCE(pc.ppi_use_count_2yr, 0) AS ppi_use_count_2yr,

  COALESCE(pf.nsaid_asa_use_flag, 0) AS nsaid_asa_use_flag,
  pd.nsaid_asa_use_days_since,
  COALESCE(pc.nsaid_asa_use_count_2yr, 0) AS nsaid_asa_use_count_2yr,

  COALESCE(pf.statin_use_flag, 0) AS statin_use_flag,
  pd.statin_use_days_since,
  COALESCE(pc.statin_use_count_2yr, 0) AS statin_use_count_2yr,

  COALESCE(pf.metformin_use_flag, 0) AS metformin_use_flag,
  pd.metformin_use_days_since,
  COALESCE(pc.metformin_use_count_2yr, 0) AS metformin_use_count_2yr,

  -- New medication category features
  COALESCE(pf.ibd_meds_flag, 0) AS ibd_meds_flag,
  pd.ibd_meds_days_since,
  COALESCE(pc.ibd_meds_count_2yr, 0) AS ibd_meds_count_2yr,

  COALESCE(pf.hemorrhoid_meds_flag, 0) AS hemorrhoid_meds_flag,
  pd.hemorrhoid_meds_days_since,
  COALESCE(pc.hemorrhoid_meds_count_2yr, 0) AS hemorrhoid_meds_count_2yr,

  COALESCE(pf.gi_bleed_meds_flag, 0) AS gi_bleed_meds_flag,
  pd.gi_bleed_meds_days_since,
  COALESCE(pc.gi_bleed_meds_count_2yr, 0) AS gi_bleed_meds_count_2yr,

  COALESCE(pf.opioid_use_flag, 0) AS opioid_use_flag,
  pd.opioid_use_days_since,
  COALESCE(pc.opioid_use_count_2yr, 0) AS opioid_use_count_2yr,

  COALESCE(pf.broad_abx_flag, 0) AS broad_abx_flag,
  pd.broad_abx_days_since,
  COALESCE(pc.broad_abx_count_2yr, 0) AS broad_abx_count_2yr,

  COALESCE(pf.hormone_therapy_flag, 0) AS hormone_therapy_flag,
  pd.hormone_therapy_days_since,
  COALESCE(pc.hormone_therapy_count_2yr, 0) AS hormone_therapy_count_2yr,

  COALESCE(pf.chemo_agents_flag, 0) AS chemo_agents_flag,
  pd.chemo_agents_days_since,
  COALESCE(pc.chemo_agents_count_2yr, 0) AS chemo_agents_count_2yr

FROM cohort c
LEFT JOIN pivot_flags pf ON c.PAT_ID = pf.PAT_ID AND c.END_DTTM = pf.END_DTTM
LEFT JOIN pivot_days pd ON c.PAT_ID = pd.PAT_ID AND c.END_DTTM = pd.END_DTTM  
LEFT JOIN pivot_counts pc ON c.PAT_ID = pc.PAT_ID AND c.END_DTTM = pc.END_DTTM;
''')

# ========================================
# CELL 8
# ========================================

# Cell 5: Validate row count matches cohort and examine medication prevalence
# Critical validation: ensures every cohort observation has medication features

# Row count validation
result = spark.sql(f"""
SELECT 
    COUNT(*) as outpatient_meds_count,
    (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as cohort_count,
    COUNT(*) - (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as diff
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds
""")

result.show()
assert result.collect()[0]['diff'] == 0, "ERROR: Row count mismatch!"
print("✓ Row count validation passed")

# Calculate prevalence for all medication categories
# This helps validate that categories are capturing expected populations
df_summary = spark.sql(f'''
SELECT 
  COUNT(*) as total_rows,
  COUNT(DISTINCT PAT_ID) as unique_patients,
  
  -- Existing categories prevalence
  ROUND(AVG(iron_use_flag), 4) as iron_prevalence,
  ROUND(AVG(ppi_use_flag), 4) as ppi_prevalence,
  ROUND(AVG(statin_use_flag), 4) as statin_prevalence,
  ROUND(AVG(laxative_use_flag), 4) as laxative_prevalence,
  ROUND(AVG(antidiarrheal_use_flag), 4) as antidiarrheal_prevalence,
  ROUND(AVG(metformin_use_flag), 4) as metformin_prevalence,
  ROUND(AVG(nsaid_asa_use_flag), 4) as nsaid_prevalence,
  ROUND(AVG(b12_or_folate_use_flag), 4) as b12_prevalence,
  ROUND(AVG(antispasmodic_use_flag), 4) as antispasmodic_prevalence,
  
  -- New categories prevalence
  ROUND(AVG(ibd_meds_flag), 4) as ibd_meds_prevalence,
  ROUND(AVG(hemorrhoid_meds_flag), 4) as hemorrhoid_prevalence,
  ROUND(AVG(gi_bleed_meds_flag), 4) as gi_bleed_prevalence,
  ROUND(AVG(opioid_use_flag), 4) as opioid_prevalence,
  ROUND(AVG(broad_abx_flag), 4) as broad_abx_prevalence,
  ROUND(AVG(hormone_therapy_flag), 4) as hormone_prevalence,
  ROUND(AVG(chemo_agents_flag), 4) as chemo_prevalence
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds
''')

print("\n=== OUTPATIENT MEDICATION PREVALENCE ===")
display(df_summary)

# Check correlation with CRC outcome for validation
print("\n=== CHECKING ASSOCIATION WITH CRC OUTCOME ===")
outcome_check = spark.sql(f'''
SELECT 
  'Laxatives' as medication_category,
  AVG(CASE WHEN m.laxative_use_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_med,
  AVG(CASE WHEN m.laxative_use_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_med
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
UNION ALL

SELECT 
  'IBD Medications' as medication_category,
  AVG(CASE WHEN m.ibd_meds_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.ibd_meds_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
UNION ALL

SELECT 
  'Hemorrhoid Meds' as medication_category,
  AVG(CASE WHEN m.hemorrhoid_meds_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.hemorrhoid_meds_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
''')

display(outcome_check)

# ========================================
# CELL 9
# ========================================

# Cell 6: Comprehensive prevalence analysis with CRC association
# Examines medication prevalence and their association with CRC outcome

df_summary = spark.sql(f'''
WITH prevalence AS (
  SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    
    -- Calculate prevalence for all categories
    ROUND(AVG(iron_use_flag), 4) as iron_prevalence,
    ROUND(AVG(ppi_use_flag), 4) as ppi_prevalence,
    ROUND(AVG(statin_use_flag), 4) as statin_prevalence,
    ROUND(AVG(laxative_use_flag), 4) as laxative_prevalence,
    ROUND(AVG(antidiarrheal_use_flag), 4) as antidiarrheal_prevalence,
    ROUND(AVG(metformin_use_flag), 4) as metformin_prevalence,
    ROUND(AVG(nsaid_asa_use_flag), 4) as nsaid_prevalence,
    ROUND(AVG(b12_or_folate_use_flag), 4) as b12_prevalence,
    ROUND(AVG(antispasmodic_use_flag), 4) as antispasmodic_prevalence,
    ROUND(AVG(ibd_meds_flag), 4) as ibd_prevalence,
    ROUND(AVG(hemorrhoid_meds_flag), 4) as hemorrhoid_prevalence,
    ROUND(AVG(gi_bleed_meds_flag), 4) as gi_bleed_prevalence,
    ROUND(AVG(opioid_use_flag), 4) as opioid_prevalence,
    ROUND(AVG(broad_abx_flag), 4) as broad_abx_prevalence,
    ROUND(AVG(hormone_therapy_flag), 4) as hormone_prevalence,
    ROUND(AVG(chemo_agents_flag), 4) as chemo_prevalence
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds
),
crc_association AS (
  SELECT 
    -- Calculate CRC rates for each medication category
    ROUND(AVG(CASE WHEN m.iron_use_flag = 1 THEN c.FUTURE_CRC_EVENT END), 5) as iron_crc_rate,
    ROUND(AVG(CASE WHEN m.laxative_use_flag = 1 THEN c.FUTURE_CRC_EVENT END), 5) as laxative_crc_rate,
    ROUND(AVG(CASE WHEN m.antidiarrheal_use_flag = 1 THEN c.FUTURE_CRC_EVENT END), 5) as antidiarrheal_crc_rate,
    ROUND(AVG(CASE WHEN m.ibd_meds_flag = 1 THEN c.FUTURE_CRC_EVENT END), 5) as ibd_crc_rate,
    ROUND(AVG(CASE WHEN m.hemorrhoid_meds_flag = 1 THEN c.FUTURE_CRC_EVENT END), 5) as hemorrhoid_crc_rate,
    ROUND(AVG(c.FUTURE_CRC_EVENT), 5) as overall_crc_rate
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT * FROM prevalence, crc_association
''')

display(df_summary)
print("\n=== KEY INSIGHTS ===")
print("Look for medications with CRC rates significantly higher than baseline")
print("These are potential risk indicators for your model")

# ========================================
# CELL 10
# ========================================

# Cell 7: Analyze temporal patterns and recency effects
# Understanding how medication timing relates to CRC risk

temporal_analysis = spark.sql(f'''
WITH recency_bands AS (
  SELECT 
    PAT_ID,
    END_DTTM,
    -- Create recency bands for key medications
    CASE 
      WHEN laxative_use_days_since IS NULL THEN 'Never'
      WHEN laxative_use_days_since <= 30 THEN '0-30 days'
      WHEN laxative_use_days_since <= 90 THEN '31-90 days'
      WHEN laxative_use_days_since <= 180 THEN '91-180 days'
      WHEN laxative_use_days_since <= 365 THEN '181-365 days'
      ELSE '365+ days'
    END as laxative_recency,
    
    CASE 
      WHEN iron_use_days_since IS NULL THEN 'Never'
      WHEN iron_use_days_since <= 30 THEN '0-30 days'
      WHEN iron_use_days_since <= 90 THEN '31-90 days'
      WHEN iron_use_days_since <= 180 THEN '91-180 days'
      WHEN iron_use_days_since <= 365 THEN '181-365 days'
      ELSE '365+ days'
    END as iron_recency,
    
    CASE 
      WHEN antidiarrheal_use_days_since IS NULL THEN 'Never'
      WHEN antidiarrheal_use_days_since <= 30 THEN '0-30 days'
      WHEN antidiarrheal_use_days_since <= 90 THEN '31-90 days'
      WHEN antidiarrheal_use_days_since <= 180 THEN '91-180 days'
      WHEN antidiarrheal_use_days_since <= 365 THEN '181-365 days'
      ELSE '365+ days'
    END as antidiarrheal_recency
    
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds
)
SELECT 
  -- Laxative recency analysis
  laxative_recency,
  COUNT(*) as n_observations,
  ROUND(AVG(c.FUTURE_CRC_EVENT), 5) as crc_rate,
  ROUND(AVG(c.FUTURE_CRC_EVENT) / (SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort), 2) as relative_risk
FROM recency_bands r
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON r.PAT_ID = c.PAT_ID AND r.END_DTTM = c.END_DTTM
GROUP BY laxative_recency
ORDER BY 
  CASE laxative_recency
    WHEN '0-30 days' THEN 1
    WHEN '31-90 days' THEN 2
    WHEN '91-180 days' THEN 3
    WHEN '181-365 days' THEN 4
    WHEN '365+ days' THEN 5
    WHEN 'Never' THEN 6
  END
''')

display(temporal_analysis)
print("\nTemporal patterns show how recency of medication use correlates with CRC risk")
print("Recent use of GI medications may indicate active symptoms")

# ========================================
# CELL 11
# ========================================

# Cell 8: Analyze medication combinations and polypharmacy patterns
# Identifies high-risk medication combinations

combo_analysis = spark.sql(f'''
WITH med_combinations AS (
  SELECT 
    PAT_ID,
    END_DTTM,
    -- Common GI symptom medication combinations
    CASE WHEN laxative_use_flag = 1 AND antidiarrheal_use_flag = 1 
         THEN 1 ELSE 0 END as alternating_bowel_pattern,
    
    CASE WHEN laxative_use_flag = 1 AND hemorrhoid_meds_flag = 1 
         THEN 1 ELSE 0 END as constipation_with_hemorrhoids,
    
    CASE WHEN iron_use_flag = 1 AND ppi_use_flag = 1 
         THEN 1 ELSE 0 END as iron_with_ppi,
    
    CASE WHEN ibd_meds_flag = 1 OR 
              (laxative_use_flag = 1 AND antidiarrheal_use_flag = 1 AND antispasmodic_use_flag = 1)
         THEN 1 ELSE 0 END as complex_gi_pattern,
    
    -- Total medication burden
    iron_use_flag + b12_or_folate_use_flag + laxative_use_flag + 
    antidiarrheal_use_flag + antispasmodic_use_flag + ppi_use_flag + 
    nsaid_asa_use_flag + statin_use_flag + metformin_use_flag +
    ibd_meds_flag + hemorrhoid_meds_flag + gi_bleed_meds_flag +
    opioid_use_flag + broad_abx_flag + hormone_therapy_flag + 
    chemo_agents_flag as total_med_categories
    
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds
)
SELECT 
  'Alternating Bowel (Lax+Antidiarr)' as pattern,
  SUM(alternating_bowel_pattern) as n_patients,
  ROUND(AVG(alternating_bowel_pattern), 4) as prevalence,
  ROUND(AVG(CASE WHEN alternating_bowel_pattern = 1 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_with,
  ROUND(AVG(CASE WHEN alternating_bowel_pattern = 0 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_without
FROM med_combinations m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM

UNION ALL

SELECT 
  'Constipation with Hemorrhoids',
  SUM(constipation_with_hemorrhoids),
  ROUND(AVG(constipation_with_hemorrhoids), 4),
  ROUND(AVG(CASE WHEN constipation_with_hemorrhoids = 1 THEN c.FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN constipation_with_hemorrhoids = 0 THEN c.FUTURE_CRC_EVENT END), 5)
FROM med_combinations m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM

UNION ALL

SELECT 
  'Iron with PPI',
  SUM(iron_with_ppi),
  ROUND(AVG(iron_with_ppi), 4),
  ROUND(AVG(CASE WHEN iron_with_ppi = 1 THEN c.FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN iron_with_ppi = 0 THEN c.FUTURE_CRC_EVENT END), 5)
FROM med_combinations m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM

UNION ALL

SELECT 
  'Complex GI Pattern',
  SUM(complex_gi_pattern),
  ROUND(AVG(complex_gi_pattern), 4),
  ROUND(AVG(CASE WHEN complex_gi_pattern = 1 THEN c.FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN complex_gi_pattern = 0 THEN c.FUTURE_CRC_EVENT END), 5)
FROM med_combinations m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
''')

display(combo_analysis)
print("\nMedication combinations reveal complex symptom patterns")
print("Alternating bowel patterns and complex GI symptoms are particularly concerning")

# ========================================
# CELL 12
# ========================================

# Cell 9: Analyze medication frequency patterns and treatment intensity
# High frequency use may indicate persistent symptoms

frequency_analysis = spark.sql(f'''
WITH frequency_categories AS (
  SELECT 
    PAT_ID,
    END_DTTM,
    
    -- Categorize laxative use frequency
    CASE 
      WHEN laxative_use_count_2yr = 0 THEN 'None'
      WHEN laxative_use_count_2yr = 1 THEN 'Single use'
      WHEN laxative_use_count_2yr BETWEEN 2 AND 5 THEN '2-5 times'
      WHEN laxative_use_count_2yr BETWEEN 6 AND 12 THEN '6-12 times'
      WHEN laxative_use_count_2yr > 12 THEN 'Chronic (>12)'
    END as laxative_frequency,
    
    -- Categorize iron supplementation frequency
    CASE 
      WHEN iron_use_count_2yr = 0 THEN 'None'
      WHEN iron_use_count_2yr = 1 THEN 'Single use'
      WHEN iron_use_count_2yr BETWEEN 2 AND 5 THEN '2-5 times'
      WHEN iron_use_count_2yr BETWEEN 6 AND 12 THEN '6-12 times'
      WHEN iron_use_count_2yr > 12 THEN 'Chronic (>12)'
    END as iron_frequency,
    
    -- Calculate treatment intensity score
    CASE 
      WHEN laxative_use_count_2yr > 12 OR antidiarrheal_use_count_2yr > 12 THEN 'High'
      WHEN laxative_use_count_2yr > 6 OR antidiarrheal_use_count_2yr > 6 THEN 'Moderate'
      WHEN laxative_use_count_2yr > 0 OR antidiarrheal_use_count_2yr > 0 THEN 'Low'
      ELSE 'None'
    END as gi_treatment_intensity
    
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds
)
SELECT 
  laxative_frequency,
  COUNT(*) as n_observations,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct_of_cohort,
  ROUND(AVG(c.FUTURE_CRC_EVENT), 5) as crc_rate,
  ROUND(AVG(c.FUTURE_CRC_EVENT) / NULLIF((SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort), 0), 2) as relative_risk
FROM frequency_categories f
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON f.PAT_ID = c.PAT_ID AND f.END_DTTM = c.END_DTTM
GROUP BY laxative_frequency
ORDER BY 
  CASE laxative_frequency
    WHEN 'None' THEN 1
    WHEN 'Single use' THEN 2
    WHEN '2-5 times' THEN 3
    WHEN '6-12 times' THEN 4
    WHEN 'Chronic (>12)' THEN 5
  END
''')

display(frequency_analysis)
print("\nFrequency analysis reveals dose-response relationship")
print("Chronic use (>12 times) shows highest risk")

# ========================================
# CELL 13
# ========================================

# Cell 10: Age-stratified medication patterns
# CRC risk varies by age, as do medication patterns

age_analysis = spark.sql(f'''
WITH age_cohorts AS (
  SELECT 
    m.*,
    c.AGE,
    c.FUTURE_CRC_EVENT,
    CASE 
      WHEN c.AGE < 50 THEN '45-49'
      WHEN c.AGE < 60 THEN '50-59'
      WHEN c.AGE < 70 THEN '60-69'
      WHEN c.AGE < 80 THEN '70-79'
      ELSE '80+'
    END as age_group
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT 
  age_group,
  COUNT(*) as n_obs,
  
  -- Prevalence by age
  ROUND(AVG(laxative_use_flag), 3) as laxative_prev,
  ROUND(AVG(iron_use_flag), 3) as iron_prev,
  ROUND(AVG(antidiarrheal_use_flag), 3) as antidiarrheal_prev,
  ROUND(AVG(ibd_meds_flag), 3) as ibd_prev,
  ROUND(AVG(hemorrhoid_meds_flag), 3) as hemorrhoid_prev,
  
  -- CRC rates by medication and age
  ROUND(AVG(CASE WHEN laxative_use_flag = 1 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_with_laxative,
  ROUND(AVG(CASE WHEN iron_use_flag = 1 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_with_iron,
  ROUND(AVG(FUTURE_CRC_EVENT), 5) as overall_crc_rate
  
FROM age_cohorts
GROUP BY age_group
ORDER BY age_group
''')

display(age_analysis)
print("\nAge-stratified analysis shows how medication-CRC associations vary by age")
print("Younger patients with GI medications may have higher relative risk")

# ========================================
# CELL 14
# ========================================

# Cell 11: Analyze potentially protective vs risk-increasing medications
# Some medications may be protective (ASA, statins) while others indicate risk

protective_analysis = spark.sql(f'''
WITH medication_effects AS (
  SELECT 
    m.*,
    c.FUTURE_CRC_EVENT,
    
    -- Potentially protective medications
    CASE WHEN nsaid_asa_use_flag = 1 OR statin_use_flag = 1 OR metformin_use_flag = 1 
         THEN 1 ELSE 0 END as has_protective_meds,
    
    -- Risk indicator medications
    CASE WHEN laxative_use_flag = 1 OR antidiarrheal_use_flag = 1 OR 
              iron_use_flag = 1 OR hemorrhoid_meds_flag = 1 OR gi_bleed_meds_flag = 1
         THEN 1 ELSE 0 END as has_risk_meds,
    
    -- Combined profile
    CASE 
      WHEN nsaid_asa_use_flag = 1 AND laxative_use_flag = 0 THEN 'ASA only'
      WHEN nsaid_asa_use_flag = 0 AND laxative_use_flag = 1 THEN 'Laxative only'
      WHEN nsaid_asa_use_flag = 1 AND laxative_use_flag = 1 THEN 'Both ASA and Laxative'
      ELSE 'Neither'
    END as asa_laxative_profile
    
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT 
  'Potentially Protective Meds' as medication_group,
  SUM(has_protective_meds) as n_with_meds,
  ROUND(AVG(has_protective_meds), 3) as prevalence,
  ROUND(AVG(CASE WHEN has_protective_meds = 1 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_with,
  ROUND(AVG(CASE WHEN has_protective_meds = 0 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_without,
  ROUND(AVG(CASE WHEN has_protective_meds = 0 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN has_protective_meds = 1 THEN FUTURE_CRC_EVENT END), 0), 2) as protective_ratio
FROM medication_effects

UNION ALL

SELECT 
  'Risk Indicator Meds',
  SUM(has_risk_meds),
  ROUND(AVG(has_risk_meds), 3),
  ROUND(AVG(CASE WHEN has_risk_meds = 1 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN has_risk_meds = 0 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN has_risk_meds = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN has_risk_meds = 0 THEN FUTURE_CRC_EVENT END), 0), 2)
FROM medication_effects

UNION ALL

SELECT 
  'ASA only',
  SUM(CASE WHEN asa_laxative_profile = 'ASA only' THEN 1 ELSE 0 END),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'ASA only' THEN 1 ELSE 0 END), 3),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'ASA only' THEN FUTURE_CRC_EVENT END), 5),
  NULL,
  NULL
FROM medication_effects

UNION ALL

SELECT 
  'Laxative only',
  SUM(CASE WHEN asa_laxative_profile = 'Laxative only' THEN 1 ELSE 0 END),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'Laxative only' THEN 1 ELSE 0 END), 3),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'Laxative only' THEN FUTURE_CRC_EVENT END), 5),
  NULL,
  NULL
FROM medication_effects

UNION ALL

SELECT 
  'Both ASA and Laxative',
  SUM(CASE WHEN asa_laxative_profile = 'Both ASA and Laxative' THEN 1 ELSE 0 END),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'Both ASA and Laxative' THEN 1 ELSE 0 END), 3),
  ROUND(AVG(CASE WHEN asa_laxative_profile = 'Both ASA and Laxative' THEN FUTURE_CRC_EVENT END), 5),
  NULL,
  NULL
FROM medication_effects
''')

display(protective_analysis)
print("\nProtective medication analysis shows complex interactions")
print("ASA may be protective, but when combined with laxatives may indicate higher risk")

# ========================================
# CELL 15
# ========================================

# Cell 12: Analyze GI medication combinations (important for CRC)
spark.sql(f'''
SELECT 
  COUNT(*) as total_patients,
  
  -- Single GI medications
  SUM(CASE WHEN iron_use_flag = 1 THEN 1 ELSE 0 END) as iron_only,
  SUM(CASE WHEN laxative_use_flag = 1 THEN 1 ELSE 0 END) as laxative_only,
  SUM(CASE WHEN antidiarrheal_use_flag = 1 THEN 1 ELSE 0 END) as antidiarrheal_only,
  SUM(CASE WHEN ppi_use_flag = 1 THEN 1 ELSE 0 END) as ppi_only,
  
  -- Key combinations
  SUM(CASE WHEN iron_use_flag = 1 AND laxative_use_flag = 1 THEN 1 ELSE 0 END) as iron_and_laxative,
  SUM(CASE WHEN iron_use_flag = 1 AND ppi_use_flag = 1 THEN 1 ELSE 0 END) as iron_and_ppi,
  SUM(CASE WHEN laxative_use_flag = 1 AND antidiarrheal_use_flag = 1 THEN 1 ELSE 0 END) as laxative_and_antidiarrheal,
  SUM(CASE WHEN ppi_use_flag = 1 AND iron_use_flag = 1 AND laxative_use_flag = 1 THEN 1 ELSE 0 END) as gi_triad,
  
  -- Any GI medication
  SUM(CASE WHEN (iron_use_flag + laxative_use_flag + antidiarrheal_use_flag + 
                 antispasmodic_use_flag + ppi_use_flag) > 0 THEN 1 ELSE 0 END) as any_gi_med,
  
  -- Multiple GI medications
  SUM(CASE WHEN (iron_use_flag + laxative_use_flag + antidiarrheal_use_flag + 
                 antispasmodic_use_flag + ppi_use_flag) >= 2 THEN 1 ELSE 0 END) as multiple_gi_meds
  
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds
''').show()

# ========================================
# CELL 16
# ========================================

# Cell 13: Convert to pandas for detailed statistics
df_spark = spark.sql('''SELECT * FROM dev.clncl_ds.herald_eda_train_outpatient_meds''')
df = df_spark.toPandas()

print("Shape:", df.shape)
print("\nNull rates:")
print(df.isnull().sum()/df.shape[0])

# ========================================
# CELL 17
# ========================================

# Cell 14: Calculate mean values for all features
df_check = df.drop(columns=['PAT_ID', 'END_DTTM'], axis=1)
print("Mean values for outpatient medication features:")
print(df_check.mean())

# ========================================
# CELL 18
# ========================================

# Cell 15: Final summary statistics
print("=" * 80)
print("OUTPATIENT MEDICATIONS FEATURE ENGINEERING SUMMARY")
print("=" * 80)

summary = spark.sql(f"""
SELECT 
    COUNT(*) as total_rows,
    COUNT(DISTINCT PAT_ID) as unique_patients,
    
    -- Core medication coverage
    ROUND(100.0 * AVG(iron_use_flag), 1) as iron_pct,
    ROUND(100.0 * AVG(laxative_use_flag), 1) as laxative_pct,
    ROUND(100.0 * AVG(ppi_use_flag), 1) as ppi_pct,
    ROUND(100.0 * AVG(statin_use_flag), 1) as statin_pct,
    
    -- Average prescription counts
    ROUND(AVG(iron_use_count_2yr), 2) as avg_iron_count,
    ROUND(AVG(laxative_use_count_2yr), 2) as avg_laxative_count,
    ROUND(AVG(ppi_use_count_2yr), 2) as avg_ppi_count,
    ROUND(AVG(statin_use_count_2yr), 2) as avg_statin_count
    
FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds
""").collect()[0]

for key, value in summary.asDict().items():
    if value is not None:
        if 'pct' in key:
            print(f"{key:30s}: {value:>10.1f}%")
        elif 'count' in key and key != 'total_rows':
            print(f"{key:30s}: {value:>10.2f}")
        else:
            print(f"{key:30s}: {value:>10,}")

print("=" * 80)
print("✓ Outpatient medications feature engineering complete")

# ========================================
# CELL 19
# ========================================

# Step 1: Calculate basic statistics using PySpark
print("Calculating feature statistics using PySpark...")

# Join with outcome data
df_spark = spark.sql("""
    SELECT m.*, c.FUTURE_CRC_EVENT
    FROM dev.clncl_ds.herald_eda_train_outpatient_meds m
    JOIN dev.clncl_ds.herald_eda_train_final_cohort c
        ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
""")

# Cache for performance
df_spark.cache()
total_rows = df_spark.count()
baseline_crc_rate = df_spark.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]

print(f"Total rows: {total_rows:,}")
print(f"Baseline CRC rate: {baseline_crc_rate:.4f}")

# ========================================
# CELL 20
# ========================================

# Step 2: Calculate Risk Ratios for Flag Features (Fast in PySpark)
flag_features = [col for col in df_spark.columns if '_flag' in col]
risk_metrics = []

for feat in flag_features:
    stats = df_spark.groupBy(feat).agg(
        F.count('*').alias('count'),
        F.avg('FUTURE_CRC_EVENT').alias('crc_rate')
    ).collect()
    
    # Parse results
    stats_dict = {row[feat]: {'count': row['count'], 'crc_rate': row['crc_rate']} for row in stats}
    
    prevalence = stats_dict.get(1, {'count': 0})['count'] / total_rows
    rate_with = stats_dict.get(1, {'crc_rate': 0})['crc_rate']
    rate_without = stats_dict.get(0, {'crc_rate': baseline_crc_rate})['crc_rate']
    risk_ratio = rate_with / (rate_without + 1e-10)
    
    # Handle edge cases for impact calculation
    if risk_ratio > 0 and prevalence > 0:
        impact = prevalence * abs(np.log2(max(risk_ratio, 1/(risk_ratio + 1e-10))))
    else:
        impact = 0
    
    risk_metrics.append({
        'feature': feat,
        'prevalence': prevalence,
        'crc_rate_with': rate_with,
        'crc_rate_without': rate_without,
        'risk_ratio': risk_ratio,
        'impact': impact
    })

risk_df = pd.DataFrame(risk_metrics).sort_values('impact', ascending=False)
print("\nTop features by impact score:")
print(risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']])

# ========================================
# CELL 21
# ========================================

# Step 3: Analyze Feature Types and Information Content
print("\nAnalyzing feature types and information content...")

# Separate features by type
flag_features = [col for col in df_spark.columns if '_flag' in col]
count_features = [col for col in df_spark.columns if '_count' in col]
days_since_features = [col for col in df_spark.columns if '_days_since' in col]

print(f"Feature types:")
print(f"  - Flag features: {len(flag_features)}")
print(f"  - Count features: {len(count_features)}")
print(f"  - Days_since features: {len(days_since_features)}")

# For days_since: These are NULL when medication never given
missing_stats = []
for feat in days_since_features:
    # NULL means never had medication - this is information, not missing data
    never_had = df_spark.filter(F.col(feat).isNull()).count() / total_rows
    
    missing_stats.append({
        'feature': feat,
        'missing_rate': never_had,  # Keep as missing_rate for compatibility
        'medication': feat.replace('_days_since', '')
    })

missing_df = pd.DataFrame(missing_stats)
print(f"\nMedications ranked by usage (from days_since nulls):")
print(missing_df.sort_values('missing_rate').head(10))

# ========================================
# CELL 22
# ========================================

# Step 4: Sample for Mutual Information
sample_fraction = min(100000 / total_rows, 1.0)
df_sample = df_spark.sampleBy("FUTURE_CRC_EVENT", 
                               fractions={0: sample_fraction, 1: 1.0},
                               seed=42).toPandas()

print(f"\nSampled {len(df_sample):,} rows for MI calculation ({len(df_sample)/total_rows*100:.1f}%)")
print(f"Sample CRC rate: {df_sample['FUTURE_CRC_EVENT'].mean():.4f}")

# Calculate MI on sample
from sklearn.feature_selection import mutual_info_classif

feature_cols = [c for c in df_sample.columns if c not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT']]
X = df_sample[feature_cols].fillna(-999)
y = df_sample['FUTURE_CRC_EVENT']

mi_scores = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop features by Mutual Information:")
print(mi_df);

# ========================================
# CELL 23
# ========================================

# Step 5: Feature Selection Logic - FIXED
# First merge all metrics to create feature_importance
feature_importance = mi_df.merge(
    risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']], 
    on='feature', 
    how='left'
).merge(
    missing_df[['feature', 'missing_rate']], 
    on='feature', 
    how='left'
)

# Fill NAs for non-flag features
feature_importance['prevalence'] = feature_importance['prevalence'].fillna(
    1 - feature_importance['missing_rate']
)
feature_importance['risk_ratio'] = feature_importance['risk_ratio'].fillna(1.0)
feature_importance['impact'] = feature_importance['impact'].fillna(0)

# Clinical knowledge filters
MUST_KEEP = [
    'hemorrhoid_meds_flag',
    'hemorrhoid_meds_days_since',  
    'iron_use_flag',
    'laxative_use_flag',
    'antidiarrheal_use_flag',
    'ppi_use_flag',
    'statin_use_flag'
]

# Remove near-zero variance features
REMOVE = ['b12_or_folate_use_flag', 'b12_or_folate_use_days_since', 
          'b12_or_folate_use_count_2yr',
          'chemo_agents_flag', 'chemo_agents_days_since', 
          'chemo_agents_count_2yr']

print(f"\nRemoving {len(REMOVE)} pre-specified near-zero variance features")
feature_importance = feature_importance[~feature_importance['feature'].isin(REMOVE)]

# ========================================
# CELL 24
# ========================================

# Step 6: Select Best Feature per Medication 
def select_optimal_features(df_importance):
    """Select best representation for each medication with balance"""
    
    selected = []
    df_importance['medication'] = df_importance['feature'].str.split('_').str[0]
    
    for med in df_importance['medication'].unique():
        med_features = df_importance[df_importance['medication'] == med]
        
        if med in ['hemorrhoid']:
            # Keep both for extreme risk
            selected.extend(med_features[
                med_features['feature'].str.contains('flag|days_since')
            ]['feature'].tolist())
            
        elif med in ['iron', 'laxative', 'antidiarrheal']:
            # Keep flag for high-risk GI
            flag_feat = med_features[med_features['feature'].str.contains('_flag')]
            if not flag_feat.empty:
                selected.append(flag_feat.iloc[0]['feature'])
                
        elif med in ['ppi', 'statin', 'metformin']:
            # For common meds, keep flag if high prevalence
            flag_feat = med_features[med_features['feature'].str.contains('_flag')]
            if not flag_feat.empty and flag_feat.iloc[0].get('prevalence', 0) > 0.05:
                selected.append(flag_feat.iloc[0]['feature'])
            else:
                # Otherwise best MI feature
                if len(med_features) > 0:
                    selected.append(med_features.nlargest(1, 'mi_score')['feature'].values[0])
        else:
            # For others, keep best MI feature
            if len(med_features) > 0:
                selected.append(med_features.nlargest(1, 'mi_score')['feature'].values[0])
    
    # Ensure must-keep features
    for feat in MUST_KEEP:
        if feat not in selected and feat in df_importance['feature'].values:
            selected.append(feat)
    
    return list(set(selected))

# CALL THE FUNCTION AND ASSIGN RESULT
selected_features = select_optimal_features(feature_importance)
print(f"\nSelected {len(selected_features)} features after optimization")

# ========================================
# CELL 25
# ========================================

# Step 7: Create Composite Features and Save
df_final = df_spark

# GI symptom composite
df_final = df_final.withColumn('gi_symptom_meds',
    F.when((F.col('laxative_use_flag') == 1) | 
           (F.col('antidiarrheal_use_flag') == 1) | 
           (F.col('antispasmodic_use_flag') == 1), 1).otherwise(0)
)

# Alternating bowel pattern
df_final = df_final.withColumn('alternating_bowel',
    F.when((F.col('laxative_use_flag') == 1) & 
           (F.col('antidiarrheal_use_flag') == 1), 1).otherwise(0)
)

# GI bleeding pattern
df_final = df_final.withColumn('gi_bleeding_pattern',
    F.when((F.col('iron_use_flag') == 1) & 
           (F.col('ppi_use_flag') == 1), 1).otherwise(0)
)

# Hemorrhoid risk score (exponential decay)
df_final = df_final.withColumn('hemorrhoid_risk_score',
    F.when(F.col('hemorrhoid_meds_days_since').isNull(), 0)
     .otherwise(30 * F.exp(-F.col('hemorrhoid_meds_days_since') / 30))
)

composite_features = ['gi_symptom_meds', 'alternating_bowel', 'gi_bleeding_pattern', 'hemorrhoid_risk_score']
selected_features.extend(composite_features)

print(f"\nAdded {len(composite_features)} composite features")
print(f"Final feature count: {len(selected_features)}")

# Print final feature list
print("\n" + "="*60)
print("FINAL SELECTED FEATURES")
print("="*60)
selected_features_sorted = sorted(list(set(selected_features)))  # Remove duplicates and sort
for i, feat in enumerate(selected_features_sorted, 1):
    # Add description for clarity
    if 'hemorrhoid' in feat:
        desc = " [EXTREME RISK]"
    elif 'iron' in feat or 'laxative' in feat or 'antidiarrheal' in feat:
        desc = " [HIGH RISK]"
    elif feat in composite_features:
        desc = " [COMPOSITE]"
    elif 'ppi' in feat or 'statin' in feat:
        desc = " [COMMON/IMPORTANT]"
    else:
        desc = ""
    print(f"{i:2d}. {feat:<35} {desc}")

# Select final columns and save
final_columns = ['PAT_ID', 'END_DTTM'] + sorted(list(set(selected_features)))
df_reduced = df_final.select(*final_columns)

# Add icd_ prefix to all columns except keys
out_med_cols = [col for col in df_reduced.columns if col not in ['PAT_ID', 'END_DTTM']]
for col in out_med_cols:
    df_reduced = df_reduced.withColumnRenamed(col, f'out_med_{col}' if not col.startswith('out_med_') else col)

# Write to final table
output_table = 'dev.clncl_ds.herald_eda_train_outpatient_meds_reduced'
df_reduced.write.mode('overwrite').saveAsTable(output_table)

print("\n" + "="*60)
print("FEATURE REDUCTION SUMMARY")
print("="*60)
print(f"Original features: 48")
print(f"Selected features: {len(set(selected_features))}")
print(f"Reduction: {(1 - len(set(selected_features))/48)*100:.1f}%")
print(f"\n✓ Reduced dataset saved to: {output_table}")

# Verify save
row_count = spark.table(output_table).count()
print(f"✓ Verified {row_count:,} rows written to table")

# ========================================
# CELL 26
# ========================================

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_eda_train_outpatient_meds_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 27
# ========================================

display(df_check)

