## MT-Bench

#### Step 1. Generate model answers to MT-bench questions
```
python gen_model_answer.py
```
We only evaluate the fisrt turn questions since most of the baseline backdoor models are instruction models. 
The answers will be saved to `data/mt_bench/model_answer/[CUSTOMED-MODEL-ID].jsonl`.

#### Step 2. Generate GPT-4 judgments
There are several options to use GPT-4 as a judge, such as pairwise winrate and single-answer grading.
In MT-bench, we recommend single-answer grading as the default mode.
This mode asks GPT-4 to grade and give a score to model's answer directly without pairwise comparison.
GPT-4 will give a score on a scale of 10.

Note that you need to **create a new environment** for generating judgments, as MT-bench only supports `openai==0.28.1` \sigh

```
export OPENAI_API_KEY=XXXXXX  # set the OpenAI API key
python gen_judgment.py 
```

The judgments will be saved to `data/mt_bench/model_judgment/gpt-4_single.jsonl`

#### Step 3. Show MT-bench scores

- Show all scores
  ```
  python show_result.py
  ```

---
