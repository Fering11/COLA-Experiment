# SEM16 Development Subset Provenance

- Source file: `data/raw/semeval2016/semeval2016-task6-trainingdata-utf-8.txt`
- Source URL: `https://raw.githubusercontent.com/emsrc/SemEval2016_T6_Stance_Detection/master/semeval2016-task6-trainingdata-utf-8.txt`
- Source repository: `emsrc/SemEval2016_T6_Stance_Detection`
- SHA-256: `768A9046732E0F22C48E0D632A04433790CBD775AA8540358A3A12ABAD587C38`
- Selection: first row for each available `(Target, Stance)` pair, sorted by target; 5 targets x 3 labels = 15 rows.
- Role: development/smoke data only. The GitHub mirror has no explicit dataset license in its repository metadata, and this file is the SemEval training split rather than the official hidden test split.
- Evaluation restriction: results on this subset must not be reported as a replication of the paper's SEM16 test-set scores.
