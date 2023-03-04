all: literature_review.pdf

literature_review.pdf: literature_review.md
	pandoc literature_review.md -o literature_review.pdf
