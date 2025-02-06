import json

# Load the results from the JSON file
with open('results.json', 'r') as f:
    results = json.load(f)

# Print the final parameter estimates for each instrument set:
for inst_set, params in results.items():
    print(f"Instruments: {inst_set}")
    print(f"  alpha_1  = {params['alpha1']}")
    print(f"  alpha_2  = {params['alpha2']}")
    print(f"  epsilon  = {params['epsilon']}")
    print()

# Now, build a LaTeX table string.
# We assume that the columns appear in the following fixed order:
instrument_order = ['N2', 'N3', 'N4', 'N5']

# Header of the table (note the use of raw string for backslashes):
latex_table = r"""\begin{table}[htbp]\centering
\newcommand{\sym}[1]{\ifmmode^{#1}\else\(^{#1}\)\fi}
\caption{GMM Regression Table}
\begin{tabular}{l*{4}{c}}
\hline\hline
Instruments           &\multicolumn{1}{c}{$N_1$ $N_2$}&\multicolumn{1}{c}{$N_1$ $N_2$ $N_3$}&\multicolumn{1}{c}{$N_1$ $N_2$ $N_3$ $N_4$}&\multicolumn{1}{c}{$N_1$ $N_2$ $N_3$ $N_1\ln N_1$}\\
\hline
"""

# We then add the rows for each parameter.
# For each row we print the point estimate on the first line and then (standard error) on the second line.
#
# For any standard error that is not available, we print a dot: (.)
def fmt(val, fmt_spec=".4f"):
    return fmt_spec % val if val is not None else "."

# --- Row for $\alpha_1$ ---
row_alpha1 = r"$\alpha_1$" + "           "
for inst in instrument_order:
    est = results[inst]['alpha1']
    row_alpha1 += "&       " + f"{est:.3f}" + "         "
row_alpha1 += r"\\"
row_alpha1_se = " " * 20  # indent for second line
for inst in instrument_order:
    se = results[inst]['se_alpha1']
    # Print se in parentheses; if missing, print (.)
    se_str = f"({se:.3f})" if se is not None else "(.)"
    row_alpha1_se += "&     " + se_str + "         "
row_alpha1_se += r"\\"
latex_table += row_alpha1 + "\n" + row_alpha1_se + "\n" + r"\hline" + "\n"

# --- Row for $\alpha_2$ ---
row_alpha2 = r"$\alpha_2$" + "           "
for inst in instrument_order:
    est = results[inst]['alpha2']
    # For demonstration, you might add significance stars for some columns. For example, we add \sym{***} if p<0.001.
    # Here we simply hard-code for illustration:
    stars = ""
    if inst != 'N1 N2':  # assume significance for columns beyond the first
        stars = r"\sym{***}"
    row_alpha2 += "&      " + f"{est:.3f}" + stars + "         "
row_alpha2 += r"\\"
row_alpha2_se = " " * 20
for inst in instrument_order:
    se = results[inst]['se_alpha2']
    se_str = f"({se:.3f})" if se is not None else "(.)"
    row_alpha2_se += "&    " + se_str + "         "
row_alpha2_se += r"\\"
latex_table += row_alpha2 + "\n" + row_alpha2_se + "\n" + r"\hline" + "\n"

# --- Row for $\varepsilon$ ---
row_epsilon = r"$\varepsilon$" + "       "
for inst in instrument_order:
    est = results[inst]['epsilon']
    stars = ""
    if inst != 'N1 N2':
        stars = r"\sym{***}"
    row_epsilon += "&      " + f"{est:.4f}" + stars + "         "
row_epsilon += r"\\"
row_epsilon_se = " " * 20
for inst in instrument_order:
    se = results[inst]['se_epsilon']
    se_str = f"({se:.4f})" if se is not None else "(.)"
    row_epsilon_se += "&  " + se_str + "         "
row_epsilon_se += r"\\"
latex_table += row_epsilon + "\n" + row_epsilon_se + "\n"

# Close off the table (ignoring J, J_df, rank for now)
latex_table += r"""\hline\hline
\multicolumn{5}{l}{\footnotesize Standard errors in parentheses}\\
\multicolumn{5}{l}{\footnotesize \sym{*} \(p<0.05\), \sym{**} \(p<0.01\), \sym{***} \(p<0.001\)}\\
\end{tabular}\label{tab:gmm_table}
\end{table}
"""

# Write the LaTeX table to a file:
with open("gmm_table.tex", "w") as f:
    f.write(latex_table)

print("LaTeX table written to gmm_table.tex")
