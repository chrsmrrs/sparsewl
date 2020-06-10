package datastructure;

import java.util.Collection;
import java.util.Set;
import java.util.Map.Entry;

public interface FeatureVector<T> {
	
	public double getValue(T feature);
	public void setValue(T feature, double value);
	public void increase(T feature, double p);
	public void increaseByOne(T feature);
	public void decreaseByOne(T feature);
	public int size();
	public double dotProduct(FeatureVector<T> v);
	public boolean equals(FeatureVector<T> v);
	
	/**
	 * Returns the set of features with a positive count.
	 */
	public Collection<T> positiveFeatures();
	
	/**
	 * Returns the set of entries with a positive count.
	 */
	// TODO rename to non-zero entries
	public Set<Entry<T, Double>> nonZeroEntries();

	/**
	 * Adds the given vector.
	 */
	public void add(FeatureVector<T> v);
	
	public boolean allZero(Collection<T> features);

}
