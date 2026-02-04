# V2_Book5_2_Medications_Inpatient
# Functional cells: 22 of 56 code cells (107 total)
# Source: V2_Book5_2_Medications_Inpatient.ipynb
# =============================================================================

# ========================================
# CELL 1
# ========================================

# ---------------------------------
# Imports and Variable Declarations
# ---------------------------------

import datetime
from dateutil.relativedelta import relativedelta
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

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
    ('STATIN_USE', 1232000017),('STATIN_USE', 1765734),('STATIN_USE', 1765928),
    ('STATIN_USE', 1765229),('STATIN_USE', 1765232),('STATIN_USE', 1765246),('STATIN_USE', 1765249),
    ('STATIN_USE', 1765260),('STATIN_USE', 1765261),('STATIN_USE', 1765262),('STATIN_USE', 1765263),('STATIN_USE', 1765264),
    ('STATIN_USE', 1754575),('STATIN_USE', 1754577),('STATIN_USE', 1754603),('STATIN_USE', 1754584),
    ('STATIN_USE', 1754588),('STATIN_USE', 1754592),('STATIN_USE', 1754593),('STATIN_USE', 1754594),('STATIN_USE', 1754595),
    ('STATIN_USE', 1765241),('STATIN_USE', 1765239),('STATIN_USE', 1765240),('STATIN_USE', 1765253),('STATIN_USE', 1765254),
    ('STATIN_USE', 1030107485),('STATIN_USE', 103103567),('STATIN_USE', 103103088),('STATIN_USE', 105100537),('STATIN_USE', 1060048301),

    -- METFORMIN USE
    ('METFORMIN_USE', 103101190),('METFORMIN_USE', 1060036201),('METFORMIN_USE', 1060040001),
    ('METFORMIN_USE', 1060000704),('METFORMIN_USE', 1060040101),('METFORMIN_USE', 1765323),
    ('METFORMIN_USE', 1765276),('METFORMIN_USE', 1765279),('METFORMIN_USE', 1765282),('METFORMIN_USE', 1765284),
    ('METFORMIN_USE', 1765289),('METFORMIN_USE', 1765292),('METFORMIN_USE', 1765320),('METFORMIN_USE', 1765328),
    ('METFORMIN_USE', 1765331),('METFORMIN_USE', 1765334),('METFORMIN_USE', 1765336),('METFORMIN_USE', 1765339)
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

# Cell 3: Create unpivoted inpatient medications table
spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds_unpivoted AS

WITH
cohort AS (
  SELECT
    CAST(PAT_ID AS STRING)          AS PAT_ID,
    END_DTTM
  FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
),

map_med AS (
  SELECT
    UPPER(TRIM(CATEGORY_KEY))       AS CATEGORY,
    CAST(MEDICATION_ID AS BIGINT)   AS MEDICATION_ID,
    UPPER(TRIM(GEN_NAME))           AS GEN_NAME
  FROM {trgt_cat}.clncl_ds.herald_eda_train_med_id_category_map
),

map_grp AS (
  SELECT
    UPPER(TRIM(CATEGORY_KEY))       AS CATEGORY,
    CAST(GROUPER_ID AS BIGINT)      AS GROUPER_ID,
    UPPER(TRIM(GROUPER_NAME))       AS GROUPER_NAME
  FROM {trgt_cat}.clncl_ds.herald_eda_train_med_grouper_category_map
),

-- Compiled membership: site's link GROUPER → MEDICATION
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

-- Inpatient MAR data (medications actually given)
-- CRITICAL: Constrained by clarity_cur data availability from 2021-07-01
orders_mar AS (
  SELECT
    CAST(ome.PAT_ID AS STRING)                    AS PAT_ID,
    CAST(ome.MEDICATION_ID AS BIGINT)             AS MEDICATION_ID,
    UPPER(TRIM(ome.GENERIC_NAME))                 AS RAW_GENERIC,
    CAST(mar.TAKEN_TIME AS TIMESTAMP)             AS TAKEN_TIME,
    CAST(ome.ORDER_MED_ID AS BIGINT)              AS ORDER_MED_ID
  FROM clarity_cur.order_med_enh ome
  JOIN prod.clarity_cur.mar_admin_info_enh mar
    ON mar.ORDER_MED_ID = ome.ORDER_MED_ID
  WHERE ome.ORDERING_MODE_C = 2                   -- Inpatient orders only
    AND mar.TAKEN_TIME IS NOT NULL
    AND DATE(mar.TAKEN_TIME) >= DATE('2021-07-01')  -- Data availability cutoff
    AND UPPER(TRIM(mar.ACTION)) IN (
      'GIVEN','PATIENT/FAMILY ADMIN','GIVEN-SEE OVERRIDE',
      'ADMIN BY ANOTHER CLINICIAN (COMMENT)','NEW BAG','BOLUS','PUSH',
      'STARTED BY ANOTHER CLINICIAN','BAG SWITCHED',
      'CLINIC SAMPLE ADMINISTERED','APPLIED','FEEDING STARTED',
      'ACKNOWLEDGED','CONTRAST GIVEN','NEW BAG-SEE OVERRIDE',
      'BOLUS FROM BAG'
    )
),

-- MedID map hits
hits_med AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    mm.CATEGORY,
    om.MEDICATION_ID,
    om.RAW_GENERIC,
    om.TAKEN_TIME
  FROM cohort c
  LEFT JOIN orders_mar om
    ON om.PAT_ID = c.PAT_ID
   AND DATE(om.TAKEN_TIME) <  c.END_DTTM
   AND DATE(om.TAKEN_TIME) >= GREATEST(
         ADD_MONTHS(c.END_DTTM, -24),
         DATE('2021-07-01')  -- Respect data availability
       )
  JOIN map_med mm
    ON mm.MEDICATION_ID = om.MEDICATION_ID
),

-- Grouper map hits
hits_grp AS (
  SELECT
    c.PAT_ID, c.END_DTTM,
    mg.CATEGORY,
    om.MEDICATION_ID,
    om.RAW_GENERIC,
    om.TAKEN_TIME
  FROM cohort c
  LEFT JOIN orders_mar om
    ON om.PAT_ID = c.PAT_ID
   AND DATE(om.TAKEN_TIME) <  c.END_DTTM
   AND DATE(om.TAKEN_TIME) >= GREATEST(
         ADD_MONTHS(c.END_DTTM, -24),
         DATE('2021-07-01')  -- Respect data availability
       )
  JOIN grp_med_members gm
    ON gm.MEDICATION_ID = om.MEDICATION_ID
  JOIN map_grp mg
    ON mg.GROUPER_ID = gm.GROUPER_ID
),

hits_all AS (
  SELECT * FROM hits_med
  UNION ALL
  SELECT * FROM hits_grp
),

ranked AS (
  SELECT
    PAT_ID,
    END_DTTM,
    CATEGORY,
    MEDICATION_ID,
    RAW_GENERIC,
    TAKEN_TIME,
    DATEDIFF(END_DTTM, DATE(TAKEN_TIME)) AS DAYS_SINCE_MED,
    ROW_NUMBER() OVER (
      PARTITION BY PAT_ID, END_DTTM, CATEGORY, MEDICATION_ID, DATE(TAKEN_TIME)
      ORDER BY TAKEN_TIME DESC
    ) AS rn
  FROM hits_all
)

SELECT
  c.PAT_ID,
  c.END_DTTM,
  r.CATEGORY,
  r.MEDICATION_ID,
  r.RAW_GENERIC,
  r.TAKEN_TIME,
  r.DAYS_SINCE_MED
FROM cohort c
LEFT JOIN ranked r
  ON r.PAT_ID   = c.PAT_ID
 AND r.END_DTTM = c.END_DTTM
WHERE r.rn = 1 OR r.rn IS NULL;
''')

# ========================================
# CELL 5
# ========================================

# Cell 4: Create final pivoted inpatient medications table using window functions
# This transforms unpivoted inpatient MAR data into modeling features
# All features prefixed with "inp_" to distinguish from outpatient

spark.sql(f'''
CREATE OR REPLACE TABLE {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds AS

WITH
  cohort AS (
    SELECT CAST(PAT_ID AS STRING) AS PAT_ID, END_DTTM
    FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort
  ),

  unpvt AS (
    SELECT CAST(PAT_ID AS STRING) AS PAT_ID,
           END_DTTM,
           UPPER(TRIM(CATEGORY)) AS CATEGORY,
           CAST(DAYS_SINCE_MED AS INT) AS DAYS_SINCE_MED
    FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds_unpivoted
    WHERE CATEGORY IS NOT NULL
  ),

  -- Use window functions to get most recent medication per category
  ranked_meds AS (
    SELECT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      DAYS_SINCE_MED,
      ROW_NUMBER() OVER (
        PARTITION BY PAT_ID, END_DTTM, CATEGORY 
        ORDER BY DAYS_SINCE_MED ASC
      ) AS rn_most_recent
    FROM unpvt
  ),

  -- Get most recent (minimum days_since) per category
  most_recent_meds AS (
    SELECT 
      PAT_ID,
      END_DTTM,
      CATEGORY,
      DAYS_SINCE_MED
    FROM ranked_meds
    WHERE rn_most_recent = 1
  ),

  -- Count medications per category using window functions
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

  -- Create binary flags using window functions
  med_flags AS (
    SELECT DISTINCT
      PAT_ID,
      END_DTTM,
      CATEGORY,
      1 AS has_med_flag
    FROM unpvt
  ),

  -- Pivot most recent days - NOTE: Using "inp_" prefix for inpatient
  pivot_days AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Original categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN DAYS_SINCE_MED END) AS inp_iron_use_days_since,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN DAYS_SINCE_MED END) AS inp_b12_or_folate_use_days_since,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN DAYS_SINCE_MED END) AS inp_laxative_use_days_since,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN DAYS_SINCE_MED END) AS inp_antidiarrheal_use_days_since,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN DAYS_SINCE_MED END) AS inp_antispasmodic_use_days_since,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN DAYS_SINCE_MED END) AS inp_ppi_use_days_since,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN DAYS_SINCE_MED END) AS inp_nsaid_asa_use_days_since,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN DAYS_SINCE_MED END) AS inp_statin_use_days_since,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN DAYS_SINCE_MED END) AS inp_metformin_use_days_since,
      -- New categories for CRC risk
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN DAYS_SINCE_MED END) AS inp_ibd_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN DAYS_SINCE_MED END) AS inp_hemorrhoid_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN DAYS_SINCE_MED END) AS inp_gi_bleed_meds_days_since,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN DAYS_SINCE_MED END) AS inp_opioid_use_days_since,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN DAYS_SINCE_MED END) AS inp_broad_abx_days_since,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN DAYS_SINCE_MED END) AS inp_hormone_therapy_days_since,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN DAYS_SINCE_MED END) AS inp_chemo_agents_days_since
    FROM most_recent_meds
    GROUP BY PAT_ID, END_DTTM
  ),

  -- Pivot counts
  pivot_counts AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Original categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN med_count END) AS inp_iron_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN med_count END) AS inp_b12_or_folate_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN med_count END) AS inp_laxative_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN med_count END) AS inp_antidiarrheal_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN med_count END) AS inp_antispasmodic_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN med_count END) AS inp_ppi_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN med_count END) AS inp_nsaid_asa_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN med_count END) AS inp_statin_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN med_count END) AS inp_metformin_use_count_2yr,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN med_count END) AS inp_ibd_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN med_count END) AS inp_hemorrhoid_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN med_count END) AS inp_gi_bleed_meds_count_2yr,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN med_count END) AS inp_opioid_use_count_2yr,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN med_count END) AS inp_broad_abx_count_2yr,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN med_count END) AS inp_hormone_therapy_count_2yr,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN med_count END) AS inp_chemo_agents_count_2yr
    FROM med_counts
    GROUP BY PAT_ID, END_DTTM
  ),

  -- Pivot flags
  pivot_flags AS (
    SELECT
      PAT_ID,
      END_DTTM,
      -- Original categories
      SUM(CASE WHEN CATEGORY = 'IRON_SUPPLEMENTATION' THEN has_med_flag END) AS inp_iron_use_flag,
      SUM(CASE WHEN CATEGORY = 'B12_FOLATE_USE' THEN has_med_flag END) AS inp_b12_or_folate_use_flag,
      SUM(CASE WHEN CATEGORY = 'LAXATIVE_USE' THEN has_med_flag END) AS inp_laxative_use_flag,
      SUM(CASE WHEN CATEGORY = 'ANTIDIARRHEAL_USE' THEN has_med_flag END) AS inp_antidiarrheal_use_flag,
      SUM(CASE WHEN CATEGORY = 'ANTISPASMODIC_USE' THEN has_med_flag END) AS inp_antispasmodic_use_flag,
      SUM(CASE WHEN CATEGORY = 'PPI_USE' THEN has_med_flag END) AS inp_ppi_use_flag,
      SUM(CASE WHEN CATEGORY = 'NSAID_ASA_USE' THEN has_med_flag END) AS inp_nsaid_asa_use_flag,
      SUM(CASE WHEN CATEGORY = 'STATIN_USE' THEN has_med_flag END) AS inp_statin_use_flag,
      SUM(CASE WHEN CATEGORY = 'METFORMIN_USE' THEN has_med_flag END) AS inp_metformin_use_flag,
      -- New categories
      SUM(CASE WHEN CATEGORY = 'IBD_MEDICATIONS' THEN has_med_flag END) AS inp_ibd_meds_flag,
      SUM(CASE WHEN CATEGORY = 'HEMORRHOID_RECTAL_MEDS' THEN has_med_flag END) AS inp_hemorrhoid_meds_flag,
      SUM(CASE WHEN CATEGORY = 'GI_BLEEDING_MEDS' THEN has_med_flag END) AS inp_gi_bleed_meds_flag,
      SUM(CASE WHEN CATEGORY = 'CHRONIC_OPIOID_USE' THEN has_med_flag END) AS inp_opioid_use_flag,
      SUM(CASE WHEN CATEGORY = 'BROAD_SPECTRUM_ANTIBIOTICS' THEN has_med_flag END) AS inp_broad_abx_flag,
      SUM(CASE WHEN CATEGORY = 'HORMONE_THERAPY' THEN has_med_flag END) AS inp_hormone_therapy_flag,
      SUM(CASE WHEN CATEGORY = 'CHEMOTHERAPY_AGENTS' THEN has_med_flag END) AS inp_chemo_agents_flag
    FROM med_flags
    GROUP BY PAT_ID, END_DTTM
  )

SELECT
  c.PAT_ID,
  c.END_DTTM,

  -- Original medication features with inp_ prefix
  COALESCE(pf.inp_iron_use_flag, 0) AS inp_iron_use_flag,
  pd.inp_iron_use_days_since,
  COALESCE(pc.inp_iron_use_count_2yr, 0) AS inp_iron_use_count_2yr,

  COALESCE(pf.inp_b12_or_folate_use_flag, 0) AS inp_b12_or_folate_use_flag,
  pd.inp_b12_or_folate_use_days_since,
  COALESCE(pc.inp_b12_or_folate_use_count_2yr, 0) AS inp_b12_or_folate_use_count_2yr,

  COALESCE(pf.inp_laxative_use_flag, 0) AS inp_laxative_use_flag,
  pd.inp_laxative_use_days_since,
  COALESCE(pc.inp_laxative_use_count_2yr, 0) AS inp_laxative_use_count_2yr,

  COALESCE(pf.inp_antidiarrheal_use_flag, 0) AS inp_antidiarrheal_use_flag,
  pd.inp_antidiarrheal_use_days_since,
  COALESCE(pc.inp_antidiarrheal_use_count_2yr, 0) AS inp_antidiarrheal_use_count_2yr,

  COALESCE(pf.inp_antispasmodic_use_flag, 0) AS inp_antispasmodic_use_flag,
  pd.inp_antispasmodic_use_days_since,
  COALESCE(pc.inp_antispasmodic_use_count_2yr, 0) AS inp_antispasmodic_use_count_2yr,

  COALESCE(pf.inp_ppi_use_flag, 0) AS inp_ppi_use_flag,
  pd.inp_ppi_use_days_since,
  COALESCE(pc.inp_ppi_use_count_2yr, 0) AS inp_ppi_use_count_2yr,

  COALESCE(pf.inp_nsaid_asa_use_flag, 0) AS inp_nsaid_asa_use_flag,
  pd.inp_nsaid_asa_use_days_since,
  COALESCE(pc.inp_nsaid_asa_use_count_2yr, 0) AS inp_nsaid_asa_use_count_2yr,

  COALESCE(pf.inp_statin_use_flag, 0) AS inp_statin_use_flag,
  pd.inp_statin_use_days_since,
  COALESCE(pc.inp_statin_use_count_2yr, 0) AS inp_statin_use_count_2yr,

  COALESCE(pf.inp_metformin_use_flag, 0) AS inp_metformin_use_flag,
  pd.inp_metformin_use_days_since,
  COALESCE(pc.inp_metformin_use_count_2yr, 0) AS inp_metformin_use_count_2yr,

  -- New medication category features with inp_ prefix
  COALESCE(pf.inp_ibd_meds_flag, 0) AS inp_ibd_meds_flag,
  pd.inp_ibd_meds_days_since,
  COALESCE(pc.inp_ibd_meds_count_2yr, 0) AS inp_ibd_meds_count_2yr,

  COALESCE(pf.inp_hemorrhoid_meds_flag, 0) AS inp_hemorrhoid_meds_flag,
  pd.inp_hemorrhoid_meds_days_since,
  COALESCE(pc.inp_hemorrhoid_meds_count_2yr, 0) AS inp_hemorrhoid_meds_count_2yr,

  COALESCE(pf.inp_gi_bleed_meds_flag, 0) AS inp_gi_bleed_meds_flag,
  pd.inp_gi_bleed_meds_days_since,
  COALESCE(pc.inp_gi_bleed_meds_count_2yr, 0) AS inp_gi_bleed_meds_count_2yr,

  COALESCE(pf.inp_opioid_use_flag, 0) AS inp_opioid_use_flag,
  pd.inp_opioid_use_days_since,
  COALESCE(pc.inp_opioid_use_count_2yr, 0) AS inp_opioid_use_count_2yr,

  COALESCE(pf.inp_broad_abx_flag, 0) AS inp_broad_abx_flag,
  pd.inp_broad_abx_days_since,
  COALESCE(pc.inp_broad_abx_count_2yr, 0) AS inp_broad_abx_count_2yr,

  COALESCE(pf.inp_hormone_therapy_flag, 0) AS inp_hormone_therapy_flag,
  pd.inp_hormone_therapy_days_since,
  COALESCE(pc.inp_hormone_therapy_count_2yr, 0) AS inp_hormone_therapy_count_2yr,

  COALESCE(pf.inp_chemo_agents_flag, 0) AS inp_chemo_agents_flag,
  pd.inp_chemo_agents_days_since,
  COALESCE(pc.inp_chemo_agents_count_2yr, 0) AS inp_chemo_agents_count_2yr

FROM cohort c
LEFT JOIN pivot_flags pf ON c.PAT_ID = pf.PAT_ID AND c.END_DTTM = pf.END_DTTM
LEFT JOIN pivot_days pd ON c.PAT_ID = pd.PAT_ID AND c.END_DTTM = pd.END_DTTM  
LEFT JOIN pivot_counts pc ON c.PAT_ID = pc.PAT_ID AND c.END_DTTM = pc.END_DTTM;
''')

# ========================================
# CELL 6
# ========================================

# Cell 5: Validate row count matches cohort and examine inpatient medication prevalence
# Important: Inpatient prevalence will be much lower than outpatient (only during hospitalizations)

# Row count validation - CRITICAL CHECK
result = spark.sql(f"""
SELECT 
    COUNT(*) as inpatient_meds_count,
    (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as cohort_count,
    COUNT(*) - (SELECT COUNT(*) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort) as diff
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds
""")

result.show()
assert result.collect()[0]['diff'] == 0, "ERROR: Row count mismatch!"
print("✓ Row count validation passed")

# Validate medication date ranges respect data constraints
date_validation = spark.sql(f'''
SELECT 
  MIN(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME))) as min_days_back,
  MAX(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME))) as max_days_back,
  PERCENTILE(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME)), 0.95) as p95_days_back,
  PERCENTILE(DATEDIFF(m.END_DTTM, DATE(u.TAKEN_TIME)), 0.05) as p05_days_back,
  MIN(DATE(u.TAKEN_TIME)) as earliest_med_date,
  MAX(DATE(u.TAKEN_TIME)) as latest_med_date
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds_unpivoted u
  ON m.PAT_ID = u.PAT_ID AND m.END_DTTM = u.END_DTTM
WHERE u.TAKEN_TIME IS NOT NULL
''')

print("\n=== MEDICATION DATE RANGE VALIDATION ===")
date_stats = date_validation.collect()[0]
print(f"Days lookback range: {date_stats['min_days_back']} to {date_stats['max_days_back']} days")
print(f"P05 to P95: {date_stats['p05_days_back']} to {date_stats['p95_days_back']} days")
print(f"Medication dates: {date_stats['earliest_med_date']} to {date_stats['latest_med_date']}")
print(f"Expected: max ~730 days (24 months), earliest ~2021-07-01")

# Verify no medications before data availability date
if date_stats['earliest_med_date'] < datetime.date(2021, 7, 1):
    print("⚠ WARNING: Found medications before 2021-07-01 data availability cutoff!")
else:
    print("✓ All medications respect 2021-07-01 data availability constraint")

# Calculate prevalence for all medication categories
df_summary = spark.sql(f'''
SELECT 
  COUNT(*) as total_rows,
  COUNT(DISTINCT PAT_ID) as unique_patients,
  
  -- Original categories prevalence
  ROUND(AVG(inp_iron_use_flag), 4) as inp_iron_prevalence,
  ROUND(AVG(inp_ppi_use_flag), 4) as inp_ppi_prevalence,
  ROUND(AVG(inp_statin_use_flag), 4) as inp_statin_prevalence,
  ROUND(AVG(inp_laxative_use_flag), 4) as inp_laxative_prevalence,
  ROUND(AVG(inp_antidiarrheal_use_flag), 4) as inp_antidiarrheal_prevalence,
  ROUND(AVG(inp_metformin_use_flag), 4) as inp_metformin_prevalence,
  ROUND(AVG(inp_nsaid_asa_use_flag), 4) as inp_nsaid_prevalence,
  ROUND(AVG(inp_b12_or_folate_use_flag), 4) as inp_b12_prevalence,
  ROUND(AVG(inp_antispasmodic_use_flag), 4) as inp_antispasmodic_prevalence,
  
  -- New categories prevalence (expect higher for acute care medications)
  ROUND(AVG(inp_ibd_meds_flag), 4) as inp_ibd_meds_prevalence,
  ROUND(AVG(inp_hemorrhoid_meds_flag), 4) as inp_hemorrhoid_prevalence,
  ROUND(AVG(inp_gi_bleed_meds_flag), 4) as inp_gi_bleed_prevalence,
  ROUND(AVG(inp_opioid_use_flag), 4) as inp_opioid_prevalence,
  ROUND(AVG(inp_broad_abx_flag), 4) as inp_broad_abx_prevalence,
  ROUND(AVG(inp_hormone_therapy_flag), 4) as inp_hormone_prevalence,
  ROUND(AVG(inp_chemo_agents_flag), 4) as inp_chemo_prevalence
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds
''')

print("\n=== INPATIENT MEDICATION PREVALENCE ===")
print("Note: Lower overall prevalence expected (only during hospitalizations)")
display(df_summary)

# Check association with CRC outcome for key inpatient medications
print("\n=== CHECKING ASSOCIATION WITH CRC OUTCOME (INPATIENT) ===")
outcome_check = spark.sql(f'''
SELECT 
  'Inp Laxatives' as medication_category,
  AVG(CASE WHEN m.inp_laxative_use_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_with_med,
  AVG(CASE WHEN m.inp_laxative_use_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END) as crc_rate_without_med
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
UNION ALL

SELECT 
  'Inp GI Bleeding Meds' as medication_category,
  AVG(CASE WHEN m.inp_gi_bleed_meds_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.inp_gi_bleed_meds_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
UNION ALL

SELECT 
  'Inp Opioids' as medication_category,
  AVG(CASE WHEN m.inp_opioid_use_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.inp_opioid_use_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM

UNION ALL

SELECT 
  'Inp Broad Spectrum Abx' as medication_category,
  AVG(CASE WHEN m.inp_broad_abx_flag = 1 THEN c.FUTURE_CRC_EVENT ELSE NULL END),
  AVG(CASE WHEN m.inp_broad_abx_flag = 0 THEN c.FUTURE_CRC_EVENT ELSE NULL END)
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
''')

display(outcome_check)

# Special validation: Check patients with any hospitalization
hosp_check = spark.sql(f'''
SELECT 
  COUNT(*) as total_observations,
  SUM(CASE WHEN 
    inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
    inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
    inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
    inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
    inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
    inp_chemo_agents_flag > 0 
    THEN 1 ELSE 0 END) as obs_with_any_inp_med,
  ROUND(100.0 * SUM(CASE WHEN 
    inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
    inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
    inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
    inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
    inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
    inp_chemo_agents_flag > 0 
    THEN 1 ELSE 0 END) / COUNT(*), 2) as pct_with_hospitalization
FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds
''')

print("\n=== HOSPITALIZATION CHECK ===")
print("Patients with any inpatient medication (proxy for hospitalization):")
display(hosp_check)

# ========================================
# CELL 7
# ========================================

# Cell 6: Comprehensive inpatient medication analysis
# Focus on hospitalization patterns and acute care indicators

inp_summary = spark.sql(f'''
WITH hospitalization_metrics AS (
  SELECT 
    PAT_ID,
    END_DTTM,
    -- Any inpatient medication indicates hospitalization
    CASE WHEN 
      inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
      inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
      inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
      inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
      inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
      inp_chemo_agents_flag > 0 
    THEN 1 ELSE 0 END as was_hospitalized,
    
    -- GI-related hospitalization
    CASE WHEN 
      inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_antispasmodic_use_flag +
      inp_gi_bleed_meds_flag + inp_hemorrhoid_meds_flag > 0
    THEN 1 ELSE 0 END as gi_related_hosp,
    
    -- Count total unique medication categories during hospitalizations
    inp_iron_use_flag + inp_ppi_use_flag + inp_statin_use_flag + 
    inp_laxative_use_flag + inp_antidiarrheal_use_flag + inp_metformin_use_flag +
    inp_nsaid_asa_use_flag + inp_b12_or_folate_use_flag + inp_antispasmodic_use_flag +
    inp_ibd_meds_flag + inp_hemorrhoid_meds_flag + inp_gi_bleed_meds_flag +
    inp_opioid_use_flag + inp_broad_abx_flag + inp_hormone_therapy_flag + 
    inp_chemo_agents_flag as inpatient_med_diversity
    
  FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds
)
SELECT 
  COUNT(*) as total_observations,
  SUM(was_hospitalized) as obs_with_hospitalization,
  ROUND(AVG(was_hospitalized), 3) as hospitalization_rate,
  SUM(gi_related_hosp) as gi_related_hospitalizations,
  ROUND(AVG(gi_related_hosp), 3) as gi_hosp_rate,
  ROUND(AVG(inpatient_med_diversity), 2) as avg_med_categories_when_hosp,
  
  -- CRC rates by hospitalization status
  ROUND(AVG(CASE WHEN h.was_hospitalized = 1 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_hospitalized,
  ROUND(AVG(CASE WHEN h.was_hospitalized = 0 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_not_hospitalized,
  ROUND(AVG(CASE WHEN h.gi_related_hosp = 1 THEN c.FUTURE_CRC_EVENT END), 5) as crc_rate_gi_hosp
  
FROM hospitalization_metrics h
JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
  ON h.PAT_ID = c.PAT_ID AND h.END_DTTM = c.END_DTTM
''')

display(inp_summary)
print("\nHospitalization is a strong risk indicator")
print("GI-related hospitalizations show particularly high CRC risk")

# ========================================
# CELL 8
# ========================================

# Cell 7: Analyze acute GI events during hospitalization
# GI bleeding, obstruction patterns, and emergency treatments

acute_gi_analysis = spark.sql(f'''
WITH acute_patterns AS (
  SELECT 
    m.*,
    c.FUTURE_CRC_EVENT,
    
    -- Acute GI bleeding pattern
    CASE WHEN inp_gi_bleed_meds_flag = 1 OR 
              (inp_iron_use_flag = 1 AND inp_ppi_use_flag = 1)
         THEN 1 ELSE 0 END as acute_gi_bleed_pattern,
    
    -- Obstruction/ileus pattern
    CASE WHEN inp_laxative_use_flag = 1 AND inp_opioid_use_flag = 1 AND inp_antispasmodic_use_flag = 1
         THEN 1 ELSE 0 END as obstruction_pattern,
    
    -- Sepsis/severe infection pattern
    CASE WHEN inp_broad_abx_flag = 1 AND inp_opioid_use_flag = 1
         THEN 1 ELSE 0 END as severe_infection_pattern,
    
    -- IBD flare pattern
    CASE WHEN inp_ibd_meds_flag = 1 OR 
              (inp_antidiarrheal_use_flag = 1 AND inp_antispasmodic_use_flag = 1)
         THEN 1 ELSE 0 END as ibd_flare_pattern
         
  FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT 
  'Acute GI Bleeding Pattern' as acute_event,
  SUM(acute_gi_bleed_pattern) as n_events,
  ROUND(AVG(acute_gi_bleed_pattern), 4) as prevalence,
  ROUND(AVG(CASE WHEN acute_gi_bleed_pattern = 1 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_with_event,
  ROUND(AVG(CASE WHEN acute_gi_bleed_pattern = 0 THEN FUTURE_CRC_EVENT END), 5) as crc_rate_without_event,
  ROUND(AVG(CASE WHEN acute_gi_bleed_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN acute_gi_bleed_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2) as risk_ratio
FROM acute_patterns

UNION ALL

SELECT 
  'Obstruction/Ileus Pattern',
  SUM(obstruction_pattern),
  ROUND(AVG(obstruction_pattern), 4),
  ROUND(AVG(CASE WHEN obstruction_pattern = 1 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN obstruction_pattern = 0 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN obstruction_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN obstruction_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2)
FROM acute_patterns

UNION ALL

SELECT 
  'Severe Infection Pattern',
  SUM(severe_infection_pattern),
  ROUND(AVG(severe_infection_pattern), 4),
  ROUND(AVG(CASE WHEN severe_infection_pattern = 1 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN severe_infection_pattern = 0 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN severe_infection_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN severe_infection_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2)
FROM acute_patterns

UNION ALL

SELECT 
  'IBD Flare Pattern',
  SUM(ibd_flare_pattern),
  ROUND(AVG(ibd_flare_pattern), 4),
  ROUND(AVG(CASE WHEN ibd_flare_pattern = 1 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN ibd_flare_pattern = 0 THEN FUTURE_CRC_EVENT END), 5),
  ROUND(AVG(CASE WHEN ibd_flare_pattern = 1 THEN FUTURE_CRC_EVENT END) / 
        NULLIF(AVG(CASE WHEN ibd_flare_pattern = 0 THEN FUTURE_CRC_EVENT END), 0), 2)
FROM acute_patterns

ORDER BY risk_ratio DESC
''')

display(acute_gi_analysis)
print("\nAcute GI events during hospitalization are strong CRC predictors")
print("Obstruction patterns show particularly high risk ratios")

# ========================================
# CELL 9
# ========================================

# Cell 8: Direct comparison of inpatient vs outpatient medication patterns
# Understanding care setting differences

setting_comparison = spark.sql(f'''
WITH combined_data AS (
  SELECT 
    o.PAT_ID,
    o.END_DTTM,
    c.FUTURE_CRC_EVENT,
    
    -- Outpatient medications
    o.laxative_use_flag as out_laxative,
    o.iron_use_flag as out_iron,
    o.antidiarrheal_use_flag as out_antidiarrheal,
    o.gi_bleed_meds_flag as out_gi_bleed,
    o.ibd_meds_flag as out_ibd,
    
    -- Inpatient medications
    i.inp_laxative_use_flag as inp_laxative,
    i.inp_iron_use_flag as inp_iron,
    i.inp_antidiarrheal_use_flag as inp_antidiarrheal,
    i.inp_gi_bleed_meds_flag as inp_gi_bleed,
    i.inp_ibd_meds_flag as inp_ibd,
    
    -- Care patterns
    CASE 
      WHEN o.laxative_use_flag = 1 AND i.inp_laxative_use_flag = 1 THEN 'Both settings'
      WHEN o.laxative_use_flag = 1 AND i.inp_laxative_use_flag = 0 THEN 'Outpatient only'
      WHEN o.laxative_use_flag = 0 AND i.inp_laxative_use_flag = 1 THEN 'Inpatient only'
      ELSE 'Neither'
    END as laxative_care_pattern,
    
    CASE 
      WHEN o.iron_use_flag = 1 AND i.inp_iron_use_flag = 1 THEN 'Both settings'
      WHEN o.iron_use_flag = 1 AND i.inp_iron_use_flag = 0 THEN 'Outpatient only'
      WHEN o.iron_use_flag = 0 AND i.inp_iron_use_flag = 1 THEN 'Inpatient only'
      ELSE 'Neither'
    END as iron_care_pattern
    
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds o
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds i
    ON o.PAT_ID = i.PAT_ID AND o.END_DTTM = i.END_DTTM
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    ON o.PAT_ID = c.PAT_ID AND o.END_DTTM = c.END_DTTM
)
SELECT 
  laxative_care_pattern as care_pattern,
  'Laxatives' as medication,
  COUNT(*) as n_observations,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct_of_cohort,
  ROUND(AVG(FUTURE_CRC_EVENT), 5) as crc_rate,
  ROUND(AVG(FUTURE_CRC_EVENT) / (SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort), 2) as relative_risk
FROM combined_data
GROUP BY laxative_care_pattern

UNION ALL

SELECT 
  iron_care_pattern,
  'Iron',
  COUNT(*),
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2),
  ROUND(AVG(FUTURE_CRC_EVENT), 5),
  ROUND(AVG(FUTURE_CRC_EVENT) / (SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort), 2)
FROM combined_data
GROUP BY iron_care_pattern

ORDER BY medication, care_pattern
''')

display(setting_comparison)
print("\nMedications in 'Both settings' indicate severe/chronic conditions")
print("Inpatient-only medications may indicate acute events")

# ========================================
# CELL 10
# ========================================

# Cell 9: Analyze temporal distance from last hospitalization
# Recent hospitalizations are strong risk indicators

hosp_recency = spark.sql(f'''
WITH hospitalization_recency AS (
  SELECT 
    m.*,
    c.FUTURE_CRC_EVENT,
    
    -- Find most recent hospitalization (minimum days since any inpatient med)
    LEAST(
      COALESCE(inp_iron_use_days_since, 9999),
      COALESCE(inp_ppi_use_days_since, 9999),
      COALESCE(inp_laxative_use_days_since, 9999),
      COALESCE(inp_antidiarrheal_use_days_since, 9999),
      COALESCE(inp_gi_bleed_meds_days_since, 9999),
      COALESCE(inp_opioid_use_days_since, 9999),
      COALESCE(inp_broad_abx_days_since, 9999)
    ) as days_since_any_hosp,
    
    -- Most recent GI-related hospitalization
    LEAST(
      COALESCE(inp_laxative_use_days_since, 9999),
      COALESCE(inp_antidiarrheal_use_days_since, 9999),
      COALESCE(inp_gi_bleed_meds_days_since, 9999),
      COALESCE(inp_hemorrhoid_meds_days_since, 9999)
    ) as days_since_gi_hosp
    
  FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c
    ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
),
recency_bands AS (
  SELECT 
    *,
    CASE 
      WHEN days_since_any_hosp = 9999 THEN 'Never hospitalized'
      WHEN days_since_any_hosp <= 30 THEN '0-30 days'
      WHEN days_since_any_hosp <= 90 THEN '31-90 days'
      WHEN days_since_any_hosp <= 180 THEN '91-180 days'
      WHEN days_since_any_hosp <= 365 THEN '181-365 days'
      ELSE '365+ days'
    END as hosp_recency_band,
    
    CASE 
      WHEN days_since_gi_hosp = 9999 THEN 'No GI hosp'
      WHEN days_since_gi_hosp <= 90 THEN '0-90 days'
      WHEN days_since_gi_hosp <= 365 THEN '91-365 days'
      ELSE '365+ days'
    END as gi_hosp_recency_band
  FROM hospitalization_recency
)
SELECT 
  hosp_recency_band,
  COUNT(*) as n_observations,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as pct_of_cohort,
  ROUND(AVG(FUTURE_CRC_EVENT), 5) as crc_rate,
  ROUND(AVG(FUTURE_CRC_EVENT) / (SELECT AVG(FUTURE_CRC_EVENT) FROM {trgt_cat}.clncl_ds.herald_eda_train_final_cohort), 2) as relative_risk
FROM recency_bands
GROUP BY hosp_recency_band
ORDER BY 
  CASE hosp_recency_band
    WHEN '0-30 days' THEN 1
    WHEN '31-90 days' THEN 2
    WHEN '91-180 days' THEN 3
    WHEN '181-365 days' THEN 4
    WHEN '365+ days' THEN 5
    WHEN 'Never hospitalized' THEN 6
  END
''')

display(hosp_recency)
print("\nRecent hospitalizations (<90 days) show highest CRC risk")
print("This temporal pattern is crucial for risk stratification")

# ========================================
# CELL 11
# ========================================

# Cell 10: Preview feature importance based on univariate analysis
# Ranks medication features by their association with CRC

feature_importance = spark.sql(f'''
WITH feature_associations AS (
  -- Calculate association metrics for each feature
  SELECT 'out_laxative' as feature, 
         AVG(laxative_use_flag) as prevalence,
         AVG(CASE WHEN laxative_use_flag = 1 THEN c.FUTURE_CRC_EVENT END) as crc_rate_with,
         AVG(CASE WHEN laxative_use_flag = 0 THEN c.FUTURE_CRC_EVENT END) as crc_rate_without
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'out_iron',
         AVG(iron_use_flag),
         AVG(CASE WHEN iron_use_flag = 1 THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN iron_use_flag = 0 THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'out_antidiarrheal',
         AVG(antidiarrheal_use_flag),
         AVG(CASE WHEN antidiarrheal_use_flag = 1 THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN antidiarrheal_use_flag = 0 THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_eda_train_outpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'inp_gi_bleed_meds',
         AVG(inp_gi_bleed_meds_flag),
         AVG(CASE WHEN inp_gi_bleed_meds_flag = 1 THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN inp_gi_bleed_meds_flag = 0 THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
  
  UNION ALL
  
  SELECT 'inp_any_hospitalization',
         AVG(CASE WHEN inp_iron_use_flag + inp_ppi_use_flag + inp_laxative_use_flag + 
                       inp_antidiarrheal_use_flag + inp_gi_bleed_meds_flag + inp_opioid_use_flag > 0 
             THEN 1 ELSE 0 END),
         AVG(CASE WHEN inp_iron_use_flag + inp_ppi_use_flag + inp_laxative_use_flag + 
                       inp_antidiarrheal_use_flag + inp_gi_bleed_meds_flag + inp_opioid_use_flag > 0 
             THEN c.FUTURE_CRC_EVENT END),
         AVG(CASE WHEN inp_iron_use_flag + inp_ppi_use_flag + inp_laxative_use_flag + 
                       inp_antidiarrheal_use_flag + inp_gi_bleed_meds_flag + inp_opioid_use_flag = 0 
             THEN c.FUTURE_CRC_EVENT END)
  FROM {trgt_cat}.clncl_ds.herald_eda_train_inpatient_meds m
  JOIN {trgt_cat}.clncl_ds.herald_eda_train_final_cohort c ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
)
SELECT 
  feature,
  ROUND(prevalence, 3) as prevalence,
  ROUND(crc_rate_with, 5) as crc_rate_with_feature,
  ROUND(crc_rate_without, 5) as crc_rate_without_feature,
  ROUND(crc_rate_with / NULLIF(crc_rate_without, 0), 2) as risk_ratio,
  ROUND(crc_rate_with - crc_rate_without, 5) as risk_difference,
  -- Simple importance score based on prevalence and risk ratio
  ROUND((prevalence * (crc_rate_with / NULLIF(crc_rate_without, 0) - 1)), 4) as importance_score
FROM feature_associations
ORDER BY importance_score DESC
''')

display(feature_importance)
print("\nFeature importance preview shows most predictive medication features")
print("Hospitalization and GI symptom medications rank highest")

# ========================================
# CELL 12
# ========================================

# Cell 11: Convert to pandas for detailed statistics
df_spark = spark.sql('''SELECT * FROM dev.clncl_ds.herald_eda_train_inpatient_meds''')
df = df_spark.toPandas()

print("Shape:", df.shape)
print("\nNull rates:")
print(df.isnull().sum()/df.shape[0])

# ========================================
# CELL 13
# ========================================

# Cell 12: Calculate mean values for all features
df_check = df.drop(columns=['PAT_ID', 'END_DTTM'], axis=1)
print("Mean values for intpatient medication features:")
print(df_check.mean())

# ========================================
# CELL 14
# ========================================

# Step 1: Load inpatient data and calculate basic statistics

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("INPATIENT MEDICATION FEATURE REDUCTION")
print("="*60)

# Join with outcome data
df_spark = spark.sql("""
    SELECT m.*, c.FUTURE_CRC_EVENT
    FROM dev.clncl_ds.herald_eda_train_inpatient_meds m
    JOIN dev.clncl_ds.herald_eda_train_final_cohort c
        ON m.PAT_ID = c.PAT_ID AND m.END_DTTM = c.END_DTTM
""")

# Check column names - they should already have inp_ prefix
print("Sample column names:")
print([col for col in df_spark.columns if col not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT']][:5])

# Cache for performance
df_spark.cache()
total_rows = df_spark.count()
baseline_crc_rate = df_spark.select(F.avg('FUTURE_CRC_EVENT')).collect()[0][0]

print(f"\nTotal rows: {total_rows:,}")
print(f"Baseline CRC rate: {baseline_crc_rate:.4f}")

# Calculate hospitalization rate (any inpatient medication)
hosp_rate = df_spark.filter(
    (F.col('inp_iron_use_flag') == 1) | 
    (F.col('inp_ppi_use_flag') == 1) | 
    (F.col('inp_laxative_use_flag') == 1) |
    (F.col('inp_opioid_use_flag') == 1)
).count() / total_rows

print(f"Hospitalization rate (any inpatient med): {hosp_rate:.1%}")

# ========================================
# CELL 15
# ========================================

# Step 2: Calculate Risk Ratios for Inpatient Flag Features

flag_features = [col for col in df_spark.columns if '_flag' in col and col.startswith('inp_')]
risk_metrics = []

print(f"\nCalculating risk ratios for {len(flag_features)} flag features...")

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
print("\nTop 10 features by impact score (prevalence × log risk ratio):")
print(risk_df[['feature', 'prevalence', 'risk_ratio', 'impact']].head(10).to_string())

# ========================================
# CELL 16
# ========================================

# Step 3: Analyze Feature Types and Information Content

print("\nAnalyzing feature types and information content...")

# Separate features by type
flag_features = [col for col in df_spark.columns if '_flag' in col and col.startswith('inp_')]
count_features = [col for col in df_spark.columns if '_count' in col and col.startswith('inp_')]
days_since_features = [col for col in df_spark.columns if '_days_since' in col and col.startswith('inp_')]

print(f"Feature types:")
print(f"  - Flag features: {len(flag_features)}")
print(f"  - Count features: {len(count_features)}")
print(f"  - Days_since features: {len(days_since_features)}")

# For days_since: Create missing_df for compatibility with Step 5
missing_stats = []
for feat in days_since_features:
    # NULL means never had medication - this is information, not missing data
    never_had = df_spark.filter(F.col(feat).isNull()).count() / total_rows
    
    missing_stats.append({
        'feature': feat,
        'missing_rate': never_had,  # Use missing_rate for consistency
        'medication': feat.replace('inp_', '').replace('_days_since', '')
    })

missing_df = pd.DataFrame(missing_stats)
print(f"\nMedications ranked by usage (from days_since nulls):")
print(missing_df.sort_values('missing_rate')[['feature', 'missing_rate']])

# ========================================
# CELL 17
# ========================================

# Step 4: Sample for Mutual Information Calculation

# Take stratified sample for MI calculation
sample_fraction = min(200000 / total_rows, 1.0)  # Larger sample for inpatient

print(f"\nSampling for MI calculation...")
df_sample = df_spark.sampleBy("FUTURE_CRC_EVENT", 
                               fractions={0: sample_fraction, 1: 1.0},  # Keep all positive cases
                               seed=42).toPandas()

print(f"Sampled {len(df_sample):,} rows ({len(df_sample)/total_rows*100:.1f}% of total)")
print(f"Sample CRC rate: {df_sample['FUTURE_CRC_EVENT'].mean():.4f}")

# Calculate MI on sample
from sklearn.feature_selection import mutual_info_classif

feature_cols = [c for c in df_sample.columns 
                if c not in ['PAT_ID', 'END_DTTM', 'FUTURE_CRC_EVENT'] 
                and c.startswith('inp_')]

print(f"Calculating MI for {len(feature_cols)} features...")
X = df_sample[feature_cols].fillna(-999)
y = df_sample['FUTURE_CRC_EVENT']

mi_scores = mutual_info_classif(X, y, discrete_features='auto', n_neighbors=3, random_state=42)
mi_df = pd.DataFrame({
    'feature': feature_cols,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop features by Mutual Information:")
print(mi_df.to_string())

# ========================================
# CELL 18
# ========================================

# Step 5: Feature Selection Logic for Inpatient 

# Merge all metrics (now using correct missing_df)
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

# Inpatient-specific critical features
MUST_KEEP = [
    'inp_gi_bleed_meds_flag',       # GI bleeding in hospital - critical
    'inp_iron_use_flag',             # Acute bleeding management
    'inp_laxative_use_flag',         # Post-op/obstruction
    'inp_opioid_use_flag',           # Surgery/pain indicator
    'inp_hemorrhoid_meds_flag',      # Rare but high risk
    'inp_gi_bleed_meds_days_since',  # Temporal pattern important
    'inp_ppi_use_flag',              # Common in hospital
    'inp_broad_abx_flag'             # Infection marker
]

# Note: Hemorrhoid medications kept despite extreme rarity (0.07% vs 0.2% outpatient)
# When hemorrhoids are treated inpatient, they likely represent severe bleeding/thrombosis
# requiring urgent care - making this a clinically significant signal despite low prevalence

# Pre-specified removals for near-zero features
REMOVE = ['inp_b12_or_folate_use_flag', 'inp_b12_or_folate_use_days_since', 
          'inp_b12_or_folate_use_count_2yr',  # Essentially never given inpatient
          'inp_chemo_agents_flag', 'inp_chemo_agents_days_since', 
          'inp_chemo_agents_count_2yr']  # Extremely rare inpatient

# Add features with extremely low signal
for _, row in feature_importance.iterrows():
    feat = row['feature']
    
    # More lenient for inpatient due to lower overall prevalence
    if row.get('missing_rate', 0) > 0.999 and feat not in MUST_KEEP:
        REMOVE.append(feat)
    
    # Remove extremely rare with low risk (adjusted for inpatient)
    if '_flag' in feat:
        # Lower threshold for inpatient
        if (row.get('prevalence', 0) < 0.00001 and row.get('risk_ratio', 1) < 3 and feat not in MUST_KEEP):
            REMOVE.append(feat)

REMOVE = list(set(REMOVE))  # Remove duplicates
print(f"\nRemoving {len(REMOVE)} near-zero signal features")
print(f"Examples of removed features: {REMOVE[:5]}")

feature_importance = feature_importance[~feature_importance['feature'].isin(REMOVE)]
print(f"Features remaining after filtering: {len(feature_importance)}")

# ========================================
# CELL 19
# ========================================

# Step 6: Select Best Feature per Medication for Inpatient

def select_optimal_inpatient_features(df_importance):
    """Select best representation for each inpatient medication"""
    
    selected = []
    
    # Extract medication name (remove inp_ prefix for grouping)
    df_importance['medication'] = df_importance['feature'].str.replace('inp_', '').str.split('_').str[0]
    
    # Group by medication
    for med in df_importance['medication'].unique():
        med_features = df_importance[df_importance['medication'] == med]
        
        # Inpatient-specific selection rules
        if med in ['hemorrhoid']:
            # Keep both flag and recency if available
            for feat in med_features['feature']:
                if '_flag' in feat or '_days_since' in feat:
                    selected.append(feat)
                    
        elif med in ['gi', 'iron', 'laxative', 'opioid']:
            # Critical acute care indicators - keep flag
            flag_features = med_features[med_features['feature'].str.contains('_flag')]
            if not flag_features.empty:
                selected.append(flag_features.nlargest(1, 'mi_score')['feature'].values[0])
            
            # Also keep gi_bleed days_since if it's gi_bleed
            if med == 'gi':
                days_features = med_features[med_features['feature'].str.contains('_days_since')]
                if not days_features.empty:
                    selected.append(days_features.iloc[0]['feature'])
                    
        elif med in ['broad', 'antidiarrheal', 'ppi']:
            # Acute event markers - keep flag only
            flag_features = med_features[med_features['feature'].str.contains('_flag')]
            if not flag_features.empty:
                selected.append(flag_features.nlargest(1, 'mi_score')['feature'].values[0])
                
        else:
            # For chronic meds (statins, metformin), keep best MI feature
            if len(med_features) > 0:
                best_feature = med_features.nlargest(1, 'mi_score')['feature'].values[0]
                selected.append(best_feature)
    
    # Ensure must-keep features are included
    for feat in MUST_KEEP:
        if feat not in selected and feat in df_importance['feature'].values:
            selected.append(feat)
    
    return list(set(selected))  # Remove duplicates

selected_features = select_optimal_inpatient_features(feature_importance)
print(f"\nSelected {len(selected_features)} features after medication-level optimization")
print("\nSample of selected features:")
for feat in sorted(selected_features)[:10]:
    print(f"  - {feat}")

# ========================================
# CELL 20
# ========================================

# Step 7: Create Inpatient-Specific Composite Features and Save

df_final = df_spark

# === INPATIENT-SPECIFIC COMPOSITE FEATURES ===

# 1. Acute GI bleeding pattern (iron + PPI during admission)
df_final = df_final.withColumn('inp_acute_gi_bleeding',
    F.when((F.col('inp_iron_use_flag') == 1) & 
           (F.col('inp_ppi_use_flag') == 1), 1).otherwise(0)
)

# 2. Obstruction/ileus pattern (laxatives + opioids)
df_final = df_final.withColumn('inp_obstruction_pattern',
    F.when((F.col('inp_laxative_use_flag') == 1) & 
           (F.col('inp_opioid_use_flag') == 1), 1).otherwise(0)
)

# 3. Severe infection/sepsis pattern (antibiotics + opioids)
df_final = df_final.withColumn('inp_severe_infection',
    F.when((F.col('inp_broad_abx_flag') == 1) & 
           (F.col('inp_opioid_use_flag') == 1), 1).otherwise(0)
)

# 4. Any hospitalization indicator (critical feature)
df_final = df_final.withColumn('inp_any_hospitalization',
    F.when((F.col('inp_iron_use_flag') == 1) | 
           (F.col('inp_ppi_use_flag') == 1) |
           (F.col('inp_laxative_use_flag') == 1) |
           (F.col('inp_opioid_use_flag') == 1) |
           (F.col('inp_broad_abx_flag') == 1), 1).otherwise(0)
)

# 5. GI-specific hospitalization
df_final = df_final.withColumn('inp_gi_hospitalization',
    F.when((F.col('inp_laxative_use_flag') == 1) | 
           (F.col('inp_antidiarrheal_use_flag') == 1) |
           (F.col('inp_gi_bleed_meds_flag') == 1), 1).otherwise(0)
)

composite_features = [
    'inp_acute_gi_bleeding',
    'inp_obstruction_pattern', 
    'inp_severe_infection',
    'inp_any_hospitalization',
    'inp_gi_hospitalization'
]

# Add composites to selected features
selected_features.extend(composite_features)
selected_features = sorted(list(set(selected_features)))  # Remove duplicates and sort

print(f"\nAdded {len(composite_features)} composite features")
print(f"Final feature count: {len(selected_features)}")

# === PRINT FINAL FEATURE LIST ===
print("\n" + "="*60)
print("FINAL SELECTED INPATIENT FEATURES")
print("="*60)

for i, feat in enumerate(selected_features, 1):
    # Add description for clarity
    if 'hemorrhoid' in feat:
        desc = " [RARE BUT EXTREME RISK]"
    elif 'gi_bleed' in feat:
        desc = " [ACUTE BLEEDING]"
    elif 'any_hospitalization' in feat:
        desc = " [KEY INDICATOR]"
    elif feat in composite_features:
        desc = " [COMPOSITE]"
    elif 'opioid' in feat:
        desc = " [PAIN/SURGERY]"
    elif 'iron' in feat or 'laxative' in feat:
        desc = " [HIGH RISK]"
    elif 'ppi' in feat or 'broad_abx' in feat:
        desc = " [COMMON ACUTE]"
    else:
        desc = ""
    print(f"{i:2d}. {feat:<40} {desc}")

# === SAVE REDUCED DATASET ===
final_columns = ['PAT_ID', 'END_DTTM'] + selected_features
df_reduced = df_final.select(*final_columns)

# Add icd_ prefix to all columns except keys
inp_med_cols = [col for col in df_reduced.columns if col not in ['PAT_ID', 'END_DTTM']]
for col in inp_med_cols:
    df_reduced = df_reduced.withColumnRenamed(col, f'inp_med_{col}' if not col.startswith('inp_med_') else col)

# Write to final table
output_table = 'dev.clncl_ds.herald_eda_train_inpatient_meds_reduced'
df_reduced.write.mode('overwrite').saveAsTable(output_table)

print("\n" + "="*60)
print("FEATURE REDUCTION SUMMARY")
print("="*60)
print(f"Original features: 48")
print(f"Selected features: {len(selected_features)}")
print(f"Reduction: {(1 - len(selected_features)/48)*100:.1f}%")
print(f"\n✓ Reduced dataset saved to: {output_table}")

# Verify save and check all columns have inp_ prefix
row_count = spark.table(output_table).count()
cols_without_prefix = [c for c in selected_features if not c.startswith('inp_')]

print(f"✓ Verified {row_count:,} rows written to table")
if cols_without_prefix:
    print(f"\n⚠ WARNING: These columns missing 'inp_' prefix: {cols_without_prefix}")
else:
    print("✓ All feature columns have 'inp_' prefix for joining")

# ========================================
# CELL 21
# ========================================

df_check_spark = spark.sql(f'select * from dev.clncl_ds.herald_eda_train_inpatient_meds_reduced')
df_check = df_check_spark.toPandas()
df_check.isnull().sum()/len(df_check)

# ========================================
# CELL 22
# ========================================

display(df_check)

