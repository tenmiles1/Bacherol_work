import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("metrics_log (1).csv")

#BLEU-1 to BLEU-4
plt.figure(figsize=(8, 5))
for i in range(1, 5):
    plt.plot(df['Epoch'], df[f'BLEU-{i}'], label=f'BLEU-{i}')
plt.xlabel("Епоха")
plt.ylabel("BLEU Score")
plt.title("BLEU-1..4 по епохах")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bleu_scores.png")
plt.close()

#METEOR + ROUGE
plt.figure(figsize=(8, 5))
plt.plot(df['Epoch'], df['METEOR'], label='METEOR', marker='o')
plt.plot(df['Epoch'], df['ROUGE-L'], label='ROUGE-L', marker='s')
plt.xlabel("Епоха")
plt.ylabel("Score")
plt.title("METEOR та ROUGE-L по епохах")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("meteor_rouge.png")
plt.close()

print("Графіки bleu_scores.png, meteor_rouge.png збережено")
