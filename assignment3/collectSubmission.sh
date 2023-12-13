#!/bin/bash
set -euo pipefail

CODE=(
	"cs682/transformer_layers.py"
	#"cs682/classifiers/rnn.py"
	"cs682/gan_pytorch.py"
)
NOTEBOOKS=(
	"Transformer_Captioning.ipynb"
	"Generative_Adversarial_Networks.ipynb"
	"StyleTransfer.ipynb"
)
PDFS=(
	"Transformer_Captioning.ipynb"
	"Generative_Adversarial_Networks.ipynb"
	"StyleTransfer.ipynb"
)

FILES=( "${CODE[@]}" "${NOTEBOOKS[@]}" )
ZIP_FILENAME="a3_code_submission.zip"
PDF_FILENAME="a3_inline_submission.pdf"

for FILE in "${FILES[@]}"
do
	if [ ! -f ${FILE} ]; then
		echo -e "${C_R}Required file ${FILE} not found, Exiting.${C_E}"
		exit 0
	fi
done

echo -e "### Zipping file ###"
rm -f ${ZIP_FILENAME}
zip -q "${ZIP_FILENAME}" -r ${NOTEBOOKS[@]} $(find . -name "*.py") $(find . -name "*.pyx") -x "makepdf.py"

echo -e "### Creating PDFs ###"
python makepdf.py --notebooks "${PDFS[@]}" --pdf_filename "${PDF_FILENAME}"

echo -e "### Done! Please submit ${ZIP_FILENAME} and ${PDF_FILENAME} to Gradescope. ###"
