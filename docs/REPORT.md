# Sepsis Early Warning Project Report

Author: Luka Andric
Date: 14 November 2026
This is a learning project. It is not a medical tool and it must not be used on real patients.

# 1. Summary

In this project I tried to predict sepsis in ICU patients. The model looks at the data of a patient every hour and gives a risk score for the next 6 hours.

At the start I already had a first version of the project. When I checked it again, I saw that its results were too good. The reason was data leakage. So I rebuilt the whole project the correct way.

Now the model can find which patients are at higher risk. Its score is 0.80. For this score, 0.5 means guessing and 1.0 means perfect. But the model has two big problems. It gives too many false alarms, because about 6 out of 10 patients without sepsis get at least one alarm. And it cannot tell when sepsis will start. I measured this with a timing score, and it is 0.51, which is the same as guessing.

I also could not repeat the numbers from my first report. When I ran my old code again on the same data, I got much worse numbers.

In the next sections I explain sepsis, the data, my mistakes, the models I tried, the results and the problems I still have.

# 2. What is sepsis

Sepsis is a very dangerous condition. It starts with an infection. The body reacts too strongly to the infection and begins to damage its own organs. If the patient does not get treatment fast, the patient can go into shock and can die.

The treatment is antibiotics, fluids and support for the organs. It works better when it starts early. In my first report I wrote that every hour of delay raises the risk of death by about 8%. I did not check this number again, but everybody agrees that earlier treatment is better.

Doctors in the ICU watch two kinds of data. The first kind is vital signs, which are measured all the time. These are heart rate, breathing rate, temperature, blood pressure and the oxygen in the blood. The second kind is blood tests, which are done only a few times per day. Examples are lactate, white blood cells, creatinine, bilirubin and platelets. High lactate means that the body does not get enough oxygen.

There is also a simple rule called SIRS. It counts four warning signs: heart rate above 90, breathing rate above 20, temperature above 38 or below 36 degrees, and abnormal white blood cells. If the patient has two or more signs, the doctor can suspect sepsis. I use this rule later to compare it with my model.

This problem is hard for a computer for three reasons. Other illnesses can look the same, so there are many false alarms. Blood tests are rare, so most data is missing. And if a system gives too many alarms, the nurses start to ignore it.

The goal of my project is simple. Every hour I want to give each patient a risk score for sepsis in the next 6 hours. Then the staff could look first at the patients with the highest score.

# 3. The data

I used the public data from the PhysioNet Challenge 2019. There is one row for each hour that a patient spent in the ICU.

The data has 40,336 patients and 1,552,210 rows. There are 43 columns. They contain vital signs, lab tests, age, gender, the hospital unit, the hours in the ICU and the sepsis label. In total 2,932 patients got sepsis, which is 7.3%. Only 1.8% of all hours are marked as sepsis. The median stay is 38 hours. The median hour when sepsis starts is hour 29.

I found three important things in the data.

First, sepsis is rare. Only 1.8% of the hours are sepsis hours. A model that always says no sepsis would be right 98% of the time. So accuracy is useless here.

![Fig 1: normal and sepsis hours (log scale) and patients](figures/fig1_class_balance.png)

Fig 1. Sepsis hours (left, log scale) and sepsis patients (right) are rare.

Second, a lot of values are missing. 27 of the 43 columns are missing more than 80% of the time. Lactate is missing 97% of the time.

![Fig 2: share of missing values per variable](figures/fig2_missingness.png)

Fig 2. Share of missing values for each variable. Red bars are missing more than 80% of the time.

Vital signs come every hour, but lab tests come in groups. You can see this for one patient in the next figure.

![Fig 3: which variables were measured in each hour for one patient](figures/fig3_measurement_pattern.png)

Fig 3. One patient. Blue means measured and grey means missing. The labs appear only in a few hours.

Third, I found a shortcut in the data. Patients without sepsis never stay longer than about 59 hours. There are 1,232 patients with a stay longer than 60 hours, and 82% of them have sepsis. A model could learn that a long stay means sepsis, and it would not learn any medicine. So I removed the hours in the ICU and the time from admission from the model.

![Fig 5: ICU stay length](figures/fig5_icu_length.png)

Fig 5. Stays of patients without sepsis stop at about 59 hours. Almost every longer stay is sepsis.

One more thing about the label. In this dataset the sepsis label starts 6 hours before the official time of sepsis. So when I say 6 hours ahead, it can be up to about 12 hours before the doctors would say it is sepsis.

# 4. My first attempt and my mistakes

My first version used a Random Forest with only 7 inputs. In the report I wrote that it finds 93% of the sepsis cases. When I looked at the code again, I found five mistakes. All of them make the model look better than it really is.

Mistake 1. I filled the missing values and shifted the values over the whole table. Because of this, values and labels of one patient went into the rows of the next patient. The fix is to do every step inside one patient.

Mistake 2. I split the rows randomly into train and test. The same patient was in both parts, so the model could recognise the patient. The fix is to split by patient.

Mistake 3. I calculated the averages on all the data. This way information from the test part went into the training. The fix is to calculate them only on the training part.

Mistake 4. My target also counted the hours when the patient already had sepsis. The model only had to notice sepsis that was already there. The fix is to remove those hours.

Mistake 5. I chose the alarm threshold on the test data. This makes the test result too good. The fix is to choose it on a separate validation part.

The first report also had some notes left in the text, it had no figures, and it said that the heart rate trend was important even though I never tested it.

After this I did a replica test. I ran the old code again, without any change, on the same data. The old numbers did not come back. The first report said the recall was about 72% at threshold 0.50 and about 93% at the best cut-off. In the replica with a random split I got an AUROC of 0.723, a recall of 8.9% at 0.50 and 54.7% at the best cut-off, and a precision of 4.3%. With a split by patient I got an AUROC of 0.600, a recall of 2.3% at 0.50 and 53.0% at the best cut-off, and a precision of 2.4%.

![Fig 11: AUROC of the old code and of the new model](figures/fig11_replica.png)

Fig 11. The old model has a score of 0.723 with a random split and 0.600 with a split by patient.

I learned two things from this test. The leak was real, because the score fell from 0.723 to 0.600 when I split by patient. And I cannot explain the numbers from my first report. Maybe they came from a different part of the data, but I cannot say this for sure.

# 5. The new method step by step

Step 1. I prepare the data for each patient alone. A missing value is filled with the last known value of the same patient. If a test was never done for this patient, the value stays empty. Nothing is copied from another patient.

Step 2. I build 88 inputs for every hour. There are 34 measurements with the last known value. There are 34 counters that show how many hours passed since the value was measured. There are 18 trend inputs, which are the average, the spread and the change over the last 6 hours for heart rate, oxygen, temperature, blood pressure and breathing. The last two inputs are age and gender.

Step 3. I do not use the shortcut columns, which are the hours in the ICU and the time from admission.

Step 4. I define the target. For every hour the question is: will sepsis start in the next 6 hours? I also tried 3 and 12 hours. I removed the hours when sepsis had already started. I also removed 706 patients whose sepsis started in the first 6 hours, because there is nothing to warn from. After this I have 39,627 patients and 1,523,513 hours, and 0.87% of the hours are positive.

Step 5. I split the data by patient into 5 folds. Every fold keeps whole patients together and has the same share of sepsis patients. Every patient is in the test part exactly once.

Step 6. I choose the alarm threshold on validation data. In each training part I keep 15% of the patients aside. The threshold is the one that catches 80% of the sepsis hours of those patients. I never use the test part to choose it.

Step 7. I measure the results in a few ways. I look at hourly scores and at patient scores. I also made a timing score. I take only the patients with sepsis and I ask if the model can tell the last 6 hours before the start from the hours 7 to 48 before the start. If the score is 0.5, it cannot.

Step 8. I wrote automatic tests so I can trust the code. There are 13 tests. They check that nothing is filled across patients, that the trend inputs never look into the future, that the targets are only before the start of sepsis, that the shortcut columns are not used, and that the web service makes the same inputs as the training code.

# 6. The models I tried

I tried four models on the same folds and the same 6 hour target. For all of them I use the AUROC score. It tells how often the model gives a higher risk to a sepsis hour than to a normal hour. A score of 0.5 is guessing and 1.0 is perfect.

The first model is the SIRS rule. It has an AUROC of 0.626 and a timing score of 0.52. It catches 47.5% of the sepsis hours and gives an alarm in 28.7% of the normal hours.

The second model is logistic regression. This is a simple model that adds up the inputs with weights. It has an AUROC of 0.725 and a timing score of 0.52. It catches 81.2% of the sepsis hours and gives an alarm in 54.1% of the normal hours.

The third model is LightGBM without the trend inputs. LightGBM is a model with many small decision trees. It has an AUROC of 0.797 and a timing score of 0.51. It catches 80.4% of the sepsis hours and gives an alarm in 37.1% of the normal hours.

The fourth model is LightGBM with the trend inputs. This is my final model. It has an AUROC of 0.801 and a timing score of 0.51. It catches 80.5% of the sepsis hours and gives an alarm in 36.1% of the normal hours.

![Fig 12: AUROC and timing score of the four models](figures/fig12_model_comparison.png)

Fig 12. On the left, LightGBM is the best at finding risky patients. On the right, all four models are close to 0.5 for timing.

What I understand from this: LightGBM is much better than the rule and the simple model at finding risky patients. I chose it because it works with missing values, it is fast on 1.5 million rows and it is usually strong on tables of numbers. The trend inputs added only 0.004 to the score, which is almost nothing. All four models have a timing score near 0.5, so a more complex model did not help here. The SIRS rule gives fewer alarms, but only because I did not tune it to catch 80%. So the alarm numbers are not a fair comparison, but the AUROC is.

I also tried other time horizons. The AUROC is almost the same for 3 hours (0.804), 6 hours (0.801) and 12 hours (0.796). The other score, AUPRC, goes up with the horizon, but only because sepsis hours are more common in the target (0.4%, 0.9% and 1.7%). So I cannot compare it between horizons.

![Fig 10: AUROC and AUPRC for 3, 6 and 12 hours](figures/fig10_horizons.png)

Fig 10. The AUROC is flat. The AUPRC follows how common the target is.

# 7. Results of the final model

The final model is LightGBM with a 6 hour target. It finds risky patients well, but it makes many alarms. When it catches 80% of the sepsis hours, only 1.9% of its alarms are correct.

These are the numbers. The AUROC is 0.801, plus or minus 0.003 between the folds. When I put all folds together and use bootstrap, the 95% range is 0.785 to 0.801. The recall is 80.5%. The precision is 1.9%, and the base rate is 0.87%. In 36.1% of the normal hours the model gives an alarm. For the sepsis patients, 86.8% get an alarm in the 6 hour window. But also 59.1% of the patients without sepsis get at least one alarm. There are 32.1 alarms per 100 normal patient hours. The timing score is 0.509. The Brier score is 0.0085, and for a model with no skill it is 0.0087.

Fig 6 shows two curves. The left one is the ROC curve and it looks good. The right one is the precision and recall curve and it shows the real problem. The precision stays very low because sepsis is rare.

![Fig 6: ROC curve and precision and recall curve](figures/fig6_roc_pr.png)

Fig 6. The red dot is the alarm threshold that I chose.

I also checked if the risk numbers are correct. They are more or less correct. When the model says 4% risk, about 4% of those hours really were sepsis hours. But the Brier score shows that the model is only a little better than always guessing the average risk.

![Fig 13: calibration curve](figures/fig13_calibration.png)

Fig 13. When the dots are near the dashed line, the risk numbers are about right.

Then I looked at which inputs are the most important. The top ones are: how long ago the oxygen setting (FiO2) was recorded, lactate, the 6 hour average of the breathing rate, FiO2, phosphate, how long ago lactate was measured, and the 6 hour average of the heart rate. Three inputs from the top 15 are counters of the hours since the last measurement. I think this shows what the doctors decided to test, and this depends on how worried they were. So it is not only the signals from the body. I did not test this idea, and the model may not work in another hospital.

![Fig 8a: SHAP summary of the top 15 inputs](figures/fig8a_shap_summary.png)

Fig 8a. Every dot is one hour. A dot on the right pushed the risk up. Red is a high value of the input, blue is a low value and grey is a missing value.

# 8. Problems and what I did

Most of my time went into finding problems. Here I write each one, also the ones I could not solve.

Problem 1. I could not repeat the old numbers. The old code gave a recall of 9% at 0.50 and 55% at the best cut-off, and not 72% and 93%. I ran the old code again on the same data and I wrote the result in the report. I do not know the reason.

Problem 2. The random split leaked information. The score of the old model fell from 0.723 to 0.600 when I split by patient. Now I split by patient everywhere.

Problem 3. The stay length was a shortcut. Stays over 60 hours are 82% sepsis. I removed the hours in the ICU and the admission time. The score is 0.801 without them and 0.825 with them.

Problem 4. The validation was very slow. One fold needed about 10 minutes because of slow text matching of the patient IDs. I changed the IDs to numbers and now a fold needs a few seconds.

Problem 5. A software conflict. When I upgraded a package, other tools on my computer stopped working. I put the old version back and made a separate environment only for this project.

Problem 6. The trend inputs did not help. The score went up only by 0.004 and the timing did not change. I wrote this as a negative result. The timing score is still 0.51.

Problem 7. I also tried the target from the benchmark, where the target is the sepsis label of the same hour. The score was 0.821 and the timing score 0.557. The timing is a little better, but still weak.

Problem 8. There are too many false alarms. The precision is only 1.9%. I did not solve this and it stays an open problem.

Problem 9. I could not test Docker, because it is not installed on this computer. I wrote the Docker file but I never ran it.

The biggest problem is that the model does not know when. Fig 7 shows, for sepsis patients, how often the model gives an alarm in each hour before sepsis starts. It is about 80% in every hour, from 48 hours before to 1 hour before. For patients without sepsis it is about 32%. So the model can separate the patients, but its alarm does not get stronger when sepsis comes closer.

![Fig 7: alarm rate and hours before the start of sepsis](figures/fig7_detection_curve.png)

Fig 7. The alarm rate does not go up when sepsis is getting closer.

Fig 9 shows one typical patient. I took the median case and not the best one. The risk stays flat during the whole stay.

![Fig 9: vital signs and risk of one typical sepsis patient](figures/fig9_patient_timeline.png)

Fig 9. A typical sepsis patient. The risk at the bottom stays flat before the start of sepsis, which is the red dotted line.

I measured this with the timing score. It is 0.51 for the final model and between 0.50 and 0.52 for all other models. I tried trend inputs and I tried the benchmark target, and neither of them fixed it.

I have some ideas why this happens, but I did not test them. The label starts early, so the 6 hour window may be before the body shows clear changes. The model may mostly use the signals about which tests were ordered. And the hourly values may hide fast changes.

# 9. Deployment and monitoring

I made a small web service (API) and a script that watches if the data changes. This is a prototype for learning.

The API works like this. You send the hourly history of one patient. It returns a risk score, an alarm with yes or no, and the threshold that it used. It refuses wrong input, for example an impossible age, an unknown variable name or the same hour two times. It makes the inputs with the same code as the training, and I wrote a test that checks that the results are the same. Every request is saved in a log file so I can use it for monitoring.

For monitoring, I use a script that compares the logged inputs with the training data. It uses the Population Stability Index, or PSI. Below 0.10 the data is stable, between 0.10 and 0.25 there is some change, and above 0.25 there is a big change. I tested it with normal data and with shifted data. It stays quiet for normal data and gives a warning for shifted data.

What is done: the 13 automatic tests pass. The API was tested in code with sample patients. The drift script was tested on simulated data. The code is saved in a local Git repository.

What is not done: I wrote the Docker file but I never built it, because Docker is not installed on this computer. I did not run the Google Colab notebook yet. There is no real traffic, no plan for regular drift reports and no plan for retraining. The code is not on GitHub yet. There was no clinical testing of any kind.

# 10. Conclusion

The project now gives honest results, but the model is not an early warning system yet. It can rank the patients by risk, with an AUROC of 0.80. But it cannot say when sepsis will start, because the timing score is 0.51, and it gives too many false alarms.

I also want to compare what my first report said with what I found now. The first report said the recall rose from about 72% to about 93%. I could not repeat this, because the old code gave 9% and 55%. The first report said the model predicts sepsis 6 hours ahead. This is not true, because the model cannot tell when sepsis starts. The first report said the heart rate trend was a critical input. I never tested it then. Now the 6 hour average of the heart rate is in the top 10, but the lab and oxygen inputs are stronger. The first report said recall means lives saved. This is wrong, because recall is only the share of the sepsis hours that the model catches. The first report said sepsis is under 5% of the data. In the data it is 7.3% of the patients and 1.8% of the hours.

What I want to do next. First, I want to find out why the timing does not work. I can try a different target, test the inputs about the hours since the last measurement, and try a model that reads the whole sequence. Second, I want to reduce the false alarms, for example by giving an alarm only when the risk keeps going up. Third, I want to build and test the Docker image and run the Colab notebook. Fourth, I want to add a small demo page and a regular drift report.

For my CV I can write this. I built an end to end machine learning pipeline for the risk of sepsis in ICU patients, with 40,000 patients and 1.5 million hours, using LightGBM. It was better than the SIRS rule and logistic regression, with an AUROC of 0.80 against 0.63 and 0.73. I found and fixed data leakage in my earlier version with patient based validation, automatic tests and ablation tests, and I reported that the model ranks patients but does not find the time of the start (timing AUROC 0.51). I also added a tested FastAPI service and PSI drift monitoring.
