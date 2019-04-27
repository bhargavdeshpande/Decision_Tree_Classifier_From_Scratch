import java.util.HashMap;

public class Mapper {
	private HashMap<String, HashMap<String, Integer>> attributeTrueMap;
	private HashMap<String, HashMap<String, Integer>> attributeFalseMap;
	
	Mapper() {}

	public HashMap<String, HashMap<String, Integer>> getAttributeTrueMap() {
		return attributeTrueMap;
	}

	public void setAttributeTrueMap(HashMap<String, HashMap<String, Integer>> attributeTrueMap) {
		this.attributeTrueMap = attributeTrueMap;
	}

	public HashMap<String, HashMap<String, Integer>> getAttributeFalseMap() {
		return attributeFalseMap;
	}

	public void setAttributeFalseMap(HashMap<String, HashMap<String, Integer>> attributeFalseMap) {
		this.attributeFalseMap = attributeFalseMap;
	}
	
}
