#!/bin/bash

# DEFAULT_RUN_DATA = {
#     'tokenizer': 'moses',
#     'ngram_length': 3,
#     'bloom_error_rate': 0.01,
#     'max_emails_generated': 50000,
#     'bin_size': 100,
#     'ratios': [round(0.1 * i, 1) for i in range(1, 6)],
#     'bin_bounds': [[i,i+50] for i in range(50, 550, 50)],
#     'use_bloom_filter': True,
#     'use_hash': False,
#     'sample_id': None,
#     'forget_method': 'frequency',
#     'use_frequency_threshold': True,
#     'use_bins': False,
#     'sample_size': 1000,
#     # 198185 (=81%) emails lower than or equal to 500 in length
#     'max_email_length': 500,
# }
sample_id=$(\
python3 -c \
'import emails;
run_data = emails.DEFAULT_RUN_DATA;
sample_id = emails.save_original_sample(run_data);
run_data["sample_id"] = sample_id;
run_data["forget_method"] = "random";
emails.save_forgotten_sample(run_data);
run_data["forget_method"] = "frequency";
emails.save_forgotten_sample(run_data);'\
| tail -n1)
printf "$sample_id\n"
i=1
for use_bloom_filter in 'true' 'false';
  do for use_hash in 'true' 'false';
    do for forget_method in 'random' 'frequency';
      do if [ $forget_method == 'frequency' ];
        then for use_frequency_threshold in 'true' 'false';
          do
            python3 -u emails.py \
                "{
                \"sample_id\":\"$sample_id\",
                \"use_bloom_filter\":$use_bloom_filter,
                \"use_hash\":$use_hash,
                \"forget_method\": \"$forget_method\",
                \"use_frequency_threshold\": $use_frequency_threshold
                }" \
              &>logs/log$i &
            let i+=1
          done
        else
          python3 -u emails.py \
              "{
              \"sample_id\":\"$sample_id\",
              \"use_bloom_filter\":$use_bloom_filter,
              \"use_hash\":$use_hash,
              \"forget_method\": \"$forget_method\"
              }" \
            &>logs/log$i &
          let i+=1
        fi
      done
    done
  done


