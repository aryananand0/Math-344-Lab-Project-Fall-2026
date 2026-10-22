# Movie Recommendation System

A project for MATH 344L (Applied Linear Algebra Lab) at USC. I use linear algebra to build a movie recommender in Python and Google Colab, working with the MovieLens 100K dataset.

## The idea

Movie ratings can be arranged as a matrix with one row per user and one column per movie. Most of that matrix is empty, since each person has only rated a small fraction of the movies. Recommending movies comes down to filling in those missing entries. Because people's tastes are driven by a handful of hidden factors (like genre preferences), the full ratings matrix should be close to low-rank, which is what makes techniques like the SVD useful here.

## Repository structure

| Folder | Contents |
|---|---|
| `notebooks/` | Colab notebooks, one per milestone |
| `src/` | Reusable Python code |
| `figures/` | Saved plots and images |

## Progress

- [x] Milestone 1: Ratings as a matrix (exploring the data and building the user-by-movie matrix)
- [ ] Milestone 2
- [ ] Milestone 3

## How to run

Open any notebook in `notebooks/` with Google Colab and choose **Runtime → Run all**. The notebooks download the dataset automatically, so no setup is needed.

## Data

This project uses the [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/) from GroupLens Research: 100,000 ratings from 943 users on 1,682 movies. The data is not stored in this repo because its license doesn't allow redistribution.

## Author

Aryan Anand

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
