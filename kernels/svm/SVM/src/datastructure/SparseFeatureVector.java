package datastructure;

import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.Map.Entry;

/**
 * HashMap based implementation of a feature vector.
 * 
 * @author xxx
 *
 * @param <T>
 */
public class SparseFeatureVector<T> implements FeatureVector<T> {

	// TODO use mutable integer!
	protected HashMap<T, Double> map;
	
	public SparseFeatureVector() {
		this.map = new HashMap<T, Double>();
	}
	
	public SparseFeatureVector(FeatureVector<T> fv) {
		this();
		for (Entry<T, Double> e : fv.nonZeroEntries()) {
			increase(e.getKey(), e.getValue());
		}
	}
	
	// TODO it would be more efficient to use a mutable integer!
	public void increase(T feature, double p) {
		double i = getValue(feature);
		setValue(feature, i+p);
	}
	
	public void increaseByOne(T feature) {
		increase(feature, 1);
	}

	public void decreaseByOne(T feature) {
		increase(feature, -1);
	}

	public double getValue(T feature) {
		Double i = map.get(feature);
		return i == null ? 0d : i;
	}
	
	@Override
	public void setValue(T feature, double value) {
		map.put(feature, value);
	}
	
	public double dotProduct(FeatureVector<T> v) {
		FeatureVector<T> u = this;
		if (v.size()<this.size()) {
			//swap
			u = v;
			v = this;
		}
		
		double i = 0;
		
		for (Map.Entry<T, Double> e : u.nonZeroEntries()) {
			i += e.getValue() * v.getValue(e.getKey());
		}
		
		return i;
	}
	
	public void add(FeatureVector<T> v) {
		for (Entry<T, Double> e : v.nonZeroEntries()) {
			increase(e.getKey(), e.getValue());
		}
	}
		
	public int size() {
		return map.size();
	}
	
	public String toString() {
		return map.toString();
	}

	/**
	 * {@inheritDoc}
	 * 
	 * Note: Do not modify this collection!
	 */
	public Collection<T> positiveFeatures() {
		return map.keySet();
	}

	/**
	 * {@inheritDoc}
	 * 
	 * Note: Do not modify this collection!
	 */
	public Set<Entry<T, Double>> nonZeroEntries() {
		return map.entrySet();
	}

	
	public boolean allZero(Collection<T> features) {
		for (T f : features) {
			if (getValue(f) != 0) return false;
		}
		return true;
	}

	@Override
	public boolean equals(FeatureVector<T> v) {
		if (this.size() != v.size()) return false;
		
		for (Map.Entry<T, Double> e : nonZeroEntries()) {
			if (e.getValue() != v.getValue(e.getKey())) {
				return false;
			}
		}
		
		return true;
	}
}
