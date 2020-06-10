package datastructure;

/**
 * Represents a pair of two elements.
 * 
 * @param <S>
 * @param <T>
 */
// TODO: allow null values and fix methods not expecting null
public class Pair<S,T> implements Comparable<Pair<S,T>> {

	protected final S first;
	protected final T second;
	
//	public Pair() {
//	}
	
	public Pair(S first, T second) {
		this.first = first;
		this.second = second;
	}

//	public void setFirst(S first) {
//		this.first = first;
//	}

	public S getFirst() {
		return first;
	}

//	public void setSecond(T second) {
//		this.second = second;
//	}

	public T getSecond() {
		return second;
	}
	
	@Override
	public boolean equals(Object obj) {
		if (obj instanceof Pair<?,?>) {
			Pair<?,?> p = (Pair<?,?>)obj;
			return first.equals(p.getFirst()) &&
			       second.equals(p.getSecond());
		} else {
			return false;
		}
	}
	
	@Override
	public int hashCode() {
		return first.hashCode()+31*second.hashCode();
	}
	
	@Override
	public String toString() {
		return "("+String.valueOf(first)+", "+String.valueOf(second)+")";
	}

	/**
	 * {@inheritDoc}
	 * Comparison is performed in element-order. Note: A ClassCastException
	 * is thrown if either S or T do no implement Comparable<S> and
	 * Comparable<T>, respectively.
	 */
	@SuppressWarnings("unchecked")
	public int compareTo(Pair<S, T> o) {
		int r = ((Comparable<S>)first).compareTo(o.first);
		return r != 0 ? r : ((Comparable<T>)second).compareTo(o.second);
	}
	
}
