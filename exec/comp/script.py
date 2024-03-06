import pandas as pd
import itertools
import os

model = 'xlm-roberta'


def get_metric(directory, file):
    result_dir = os.path.join(directory, 'results/')
    csv_dir = os.listdir(result_dir)[0]
    csv_path = os.path.join(result_dir, csv_dir)
    csv_file = os.path.join(csv_path, file)
    if os.path.exists(csv_file):
        with open(csv_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1]
                score = float(last_line.split(',')[1].strip())
                return score
    return None


if __name__ == '__main__':
    # Define parameter values
    epochs = [1, 2, 3]
    batch = [8, 16, 32]
    max_val = [128]#, 128]
    lr = ['1e-5', '2e-5', '3e-5', '4e-5']
    


    # Generate all combinations
    combinations = list(itertools.product([model], epochs, batch, max_val, lr))

    # Create a DataFrame from combinations
    df = pd.DataFrame(combinations, columns=['model', 'epochs', 'batch', 'max_seq_length', 'lr'])

    # Populate 'f1 macro' column
    f1_macro_scores = []
    f1_macro_relevant_scores = []

    f1_micro_scores = []
    f1_micro_relevant_scores = []
    
    f1_macro_claim_scores = []
    f1_macro_premise_scores = []
    
    f1_micro_claim_scores = []
    f1_micro_premise_scores = []

    accuracy_scores = []
    for index, row in df.iterrows():
        directory_name = f"{model}-epoch={int(row['epochs'])}-batch={int(row['batch'])}-seq-len={int(row['max_seq_length'])}-lr={row['lr']}"
        f1_macro = get_metric(directory_name, f'{model}_seq-tag_eval_f1_score_macro.csv')
        f1_macro_relevant = get_metric(directory_name, f'{model}_seq-tag_eval_relevant_f1_score_macro.csv')

        f1_micro = get_metric(directory_name, f'{model}_seq-tag_eval_f1_score_micro.csv')
        f1_micro_relevant = get_metric(directory_name, f'{model}_seq-tag_eval_relevant_f1_score_micro.csv')

        accuracy = get_metric(directory_name, f'{model}_seq-tag_eval_accuracy.csv')
        
        f1_macro_claim = get_metric(directory_name, f'{model}_seq-tag_eval_f1_score_macro_Claim.csv')
        f1_macro_premise = get_metric(directory_name, f'{model}_seq-tag_eval_f1_score_macro_Premise.csv')
        
        f1_micro_claim = get_metric(directory_name, f'{model}_seq-tag_eval_f1_score_micro_Claim.csv')
        f1_micro_premise = get_metric(directory_name, f'{model}_seq-tag_eval_f1_score_micro_Premise.csv')

        f1_macro_scores.append(f1_macro)
        f1_micro_scores.append(f1_micro)
        f1_macro_relevant_scores.append(f1_macro_relevant)
        f1_micro_relevant_scores.append(f1_micro_relevant)
        f1_macro_claim_scores.append(f1_macro_claim)
        f1_macro_premise_scores.append(f1_macro_premise)
        f1_micro_claim_scores.append(f1_micro_claim)
        f1_micro_premise_scores.append(f1_micro_premise)
        accuracy_scores.append(accuracy)
        

    df['f1 macro'] = f1_macro_scores
    df['f1 micro'] = f1_micro_scores
    df['accuracy'] = accuracy_scores
    df['f1 macro relevant'] = f1_macro_relevant_scores
    df['f1 micro relevant'] = f1_micro_relevant_scores
    df['f1 macro claim'] = f1_macro_claim_scores
    df['f1 macro premise'] = f1_macro_premise_scores
    df['f1 micro claim'] = f1_micro_claim_scores
    df['f1 micro premise'] = f1_micro_premise_scores
    

    # Export DataFrame to Excel
    df.to_excel(f'{model}_seq_tag_results.xlsx', index=False)
    print(f'{model}_seq_tag_results.xlsx created.')
