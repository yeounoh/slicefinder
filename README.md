# Slice Finder
Data slicing is important in ML systems, in that trained model is better understood and validated in subsets of the data (e.g., model performs well on the entire dataset, but fails miserably for a particular subset of the data). In practice, it is often done by human curators who know important data slices beforehand (e.g., 'gender=female' ^ 'age>=18' ^ 'country=US' data slice); however, curated slices are not always available.

Automatic data slicing poses a couple of challenges: One, the search space is exponentially large and arbitrarily small slices with high ML performance metrics are not necessarily of interest--interpretability of slice is another issue that automatic slicing tool should consider.

## Slice Finder Test Cases
```python
python slice_finder_test.py
```
