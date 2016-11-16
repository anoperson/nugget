Steps to run for the evaluation:

1. Run the java code to convert the train, valid and test data to the conll format (test data is the data provided by NIST for evaluation) -> train.txt, valid.txt, test.txt

2. Run processNuggetData.py taking the three file from step 1 to generate the *.pkl file

3. Run evaluate.py to generate the valid and test files for realis (we use the golden data for validation here to choose the best iteration)

4. Run createRealisDataset.py using the pkl file as input to generate the golden files for realis -> train.realis, valid.realis, test.realis

5. Move the valid.realis and test.realis files from golden data to some folder and put the valid.pred(best-interation).id and test.pred(best-interation).id files (from step 3) to the folder 'realisData'. Change their name to test.realis and valid.realis

6. Run evaluateRealis.py to generate valid and test files for coreference (using gold data for validation to choose the best iteration)

7. Put the train.realis file (from step 4) in the 'corefData' folder. Also, put the files valid.realis(best-interation).id and test.pred(best-interation).id (from step 6) to 'corefData'. Change their names to train.coref, test.coref, valid.coref

8. Run coref_evaluate.py (using golden data for validation to choose the best iteration)

9. Using the file test.coref(best-iteration) (from step 8) as the submission file