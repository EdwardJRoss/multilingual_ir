all: docs/literature_review.pdf

docs/literature_review.pdf: docs/literature_review.md
	pandoc docs/literature_review.md -o docs/literature_review.pdf
