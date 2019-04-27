import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

class TreeNode {
	
	String name;
	String value;
	
	List<TreeNode> children;
	HashMap<String, TreeNode> edgeMap;
	
	public TreeNode(String name, String value) {
		this.name = name;
		this.value = value;
		children = new ArrayList<>();
		edgeMap = new HashMap<>();
	}
	
}


public class DecisionTreeClassifier {
	
	List<List<String>> trainData;
	HashMap<String, Integer> positionIndexMap;
	
	public void runAlgorithm(String trainFile, String testFile, String targetAttribute, int metric) {
		
		trainData = parseData(trainFile);
		List<List<String>> testData = parseData(testFile);
		positionIndexMap = getPositionIndexMap(trainData.get(0));
		// Removing features list
		trainData.remove(0);
		testData.remove(0);
		
		// Creating attributeList
		HashSet<String> attributes = new HashSet<>();
		for(Map.Entry<String, Integer> m: positionIndexMap.entrySet()) {
			attributes.add(m.getKey());
		}
		attributes.remove(targetAttribute);
				
		//For first call, all elements will be considered	
		HashSet<Integer> elements = new HashSet<>();
		for (int index=0; index<trainData.size(); index++) {
			elements.add(index);
		}
		
		//Constructing the decision tree
		TreeNode root = DecisionTreeLearning(elements, attributes, null, targetAttribute, metric);
		
		// logic to classify test Data
		List<String> testResponseList = new ArrayList<>();
		for (List<String> testList : testData) {
			String responseFromClassifier = classify(root, testList);
			testResponseList.add(responseFromClassifier);
		}
		
		// Generating confusion matrix
		int truePositive = 0;
		int falsePositive = 0;
		int trueNegative = 0;
		int falseNegative = 0;
		
		for (int index = 0; index< testData.size(); index++) {
			String actualResponse = testData.get(index).get(positionIndexMap.get(targetAttribute));
			String classifierResponse = testResponseList.get(index);
			
			// True Positive
			if ("1".equals(actualResponse) && "1".equals(classifierResponse)) {
				truePositive++;
			}
			
			// True Negative
			if ("0".equals(actualResponse) && "0".equals(classifierResponse)) {
				trueNegative++;
			}
			
			// False Negative
			if ("1".equals(actualResponse) && "0".equals(classifierResponse)) {
				falseNegative++;
			}					
						
			// False Positive
			if ("0".equals(actualResponse) && "1".equals(classifierResponse)) {
				falsePositive++;
			}
			
		}
		
		float accuracy = (float)(truePositive+trueNegative)/testData.size(); 
		
		System.out.println("Accuracy: "+accuracy*100+ "%");
		System.out.println("True Positive Count: "+truePositive);
		System.out.println("True Negative Count: "+trueNegative);
		System.out.println("False Negative Count: "+falseNegative);
		System.out.println("False Positive Count: "+falsePositive);
		System.out.println();
		
		System.out.println("Listing all braches of the tree - From root to leaf");
		// Printing all tree branches from root to leaaves
		List<List<String>> branches = new ArrayList<>();
		
		branches = printNode(branches, root, new ArrayList<String>());
		
		int index = 1;
		for (List<String>branch : branches) {
			
			System.out.println("Branch: "+index);
			for (String element: branch) {
				System.out.print(element+" ");
			}
			index++;
			System.out.println();
		}
		
	}
	
	private List<List<String>> printNode(List<List<String>> branches, TreeNode node, List<String> tempList) {
		if ("leaf".equals(node.name)) {
			List<String> listToAdd = new ArrayList<>();
			listToAdd.addAll(tempList);
			listToAdd.add(" - 'Response = "+ node.value+"'");
			branches.add(listToAdd);
		
		} else {
			for (Map.Entry<String, TreeNode> m : node.edgeMap.entrySet()) {
				tempList.add(node.name+"="+m.getKey());
				printNode(branches, m.getValue(), tempList);
				tempList.remove(node.name+"="+m.getKey());
			}
		}
		return branches;
	}
	
	private TreeNode DecisionTreeLearning(HashSet<Integer> elements, HashSet<String> attributes, HashSet<Integer> parentElements, String targetAttribute, int metric) {
		
		if (elements.size() == 0) {
			int[] trueFalsePair = getTrueFalseCount(parentElements, targetAttribute);
			int trueCount = trueFalsePair[0];
			int falseCount = trueFalsePair[1];
			return getPrularityClassification(trueCount, falseCount);
		}
		
		int[] trueFalsePair = getTrueFalseCount(elements, targetAttribute);
		int trueCount = trueFalsePair[0];
		int falseCount = trueFalsePair[1];
		
		if (attributes.size() == 0) {
			return getPrularityClassification(trueCount, falseCount);
		}
		
		if (isSameClassification (trueCount, falseCount)) {
			return trueCount == 0 ? new TreeNode("leaf","0") : new TreeNode("leaf","1");
		} else {
			
			// Mapping data
			Mapper mapper = mapTrainData(elements, targetAttribute);
				
			String currentMostImpVar = "";
			// Metric 0: Information gain
			if (metric == 0) {
				// Finding Information gain of all variables
				HashMap<String, Double> informationGainMap = calculateInformationGain(trueCount, falseCount, attributes, mapper.getAttributeTrueMap(), mapper.getAttributeFalseMap());		
				// Finding most important attribute
				currentMostImpVar = getMostImportantVariable(informationGainMap);
				
			} else {
				currentMostImpVar = getImpVarByAlternative(attributes, mapper.getAttributeTrueMap(), mapper.getAttributeFalseMap());
			}
			// separating elements based on variable value/type
			HashMap<String, HashSet<Integer>> TypeIndexMap = new HashMap<>();
			int currentMostImpVarIndex = positionIndexMap.get(currentMostImpVar);
			for (int index: elements) {
				if (TypeIndexMap.containsKey(trainData.get(index).get(currentMostImpVarIndex))) {
					TypeIndexMap.get(trainData.get(index).get(currentMostImpVarIndex)).add(index);
				} else {
					TypeIndexMap.put(trainData.get(index).get(currentMostImpVarIndex), new HashSet<>());
				}
			}
			
			// Removing most important attribute from attribute set
			attributes.remove(currentMostImpVar);
			
			TreeNode parentNode = new TreeNode(currentMostImpVar, null);
			
			// Calling for each type/value of variable
			for(Map.Entry<String, HashSet<Integer>> m: TypeIndexMap.entrySet()) {
				TreeNode node = DecisionTreeLearning(m.getValue(), attributes, elements, targetAttribute, metric);
				parentNode.edgeMap.put(m.getKey(), node);
				parentNode.children.add(node);
			}
			
			return parentNode;
		}
		
	}
	
	
	private boolean isSameClassification(int trueCount, int falseCount) {
		if(trueCount != falseCount && (trueCount == 0 || falseCount == 0)) {
			return true;
		}
		return false;
	}

	private List<List<String>> parseData(String location) {
		List<List<String>> trainData = new ArrayList<>();
		try {
			BufferedReader br = new BufferedReader(new FileReader(location));
			String line = "";
			while((line = br.readLine())!= null) {
				String[] values_per_line = line.split(",");
				List<String> tempList = new ArrayList<>();
				tempList.addAll(Arrays.asList(values_per_line));
				trainData.add(tempList);
			}
		} catch (Exception e) {
			e.printStackTrace();
		}		
	return trainData;
	}
	
	private HashMap<String, Integer> getPositionIndexMap(List<String> featuresList) {
		HashMap<String, Integer> positionMap = new HashMap<>();
		int index = 0;
		for (String featureName: featuresList) {
			positionMap.put(featureName, index);
			index++;
		}
		return positionMap;
	}
	
	private Mapper mapTrainData(HashSet<Integer> elements, String targetAttribute) {
		HashMap<String, HashMap<String, Integer>> attributeTrueMap = new HashMap<>();
		HashMap<String, HashMap<String, Integer>> attributeFalseMap = new HashMap<>();
		HashSet<String> attributeSet = new HashSet<>();
		
		for (int index : elements) {
			List<String> tempList = trainData.get(index);
			String outcome = tempList.get(positionIndexMap.get(targetAttribute));
			
			for (Map.Entry<String, Integer> m: positionIndexMap.entrySet()) {
				
				attributeSet.add(m.getKey());
				
				String featureValue = tempList.get(m.getValue());
				if ("1".equals(outcome)) {
					if (attributeTrueMap.containsKey(m.getKey())) {						
						int innerMapValue = 1;
						if (attributeTrueMap.get(m.getKey()).containsKey(featureValue)) {
							innerMapValue = attributeTrueMap.get(m.getKey()).get(featureValue)+1;
						} 
						attributeTrueMap.get(m.getKey()).put(featureValue, innerMapValue);
					} else {
						HashMap<String, Integer> innerMap = new HashMap<>();
						innerMap.put(featureValue, 1);
						attributeTrueMap.put(m.getKey(), innerMap);
					}
				} else {
					if (attributeFalseMap.containsKey(m.getKey())) {
						int innerMapValue = 1;
						if (attributeFalseMap.get(m.getKey()).containsKey(featureValue)) {
							innerMapValue = attributeFalseMap.get(m.getKey()).get(featureValue)+1;
						} 
						attributeFalseMap.get(m.getKey()).put(featureValue, innerMapValue);
					} else {
						HashMap<String, Integer> innerMap = new HashMap<>();
						innerMap.put(featureValue, 1);
						attributeFalseMap.put(m.getKey(), innerMap);
					}
				}
			}
					
		}
		Mapper mapper = new Mapper();
		mapper.setAttributeTrueMap(attributeTrueMap);
		mapper.setAttributeFalseMap(attributeFalseMap);
		return mapper;
	}
	
	private HashMap<String, Double> calculateInformationGain(int trueCount, int falseCount, HashSet<String> attributeSet,
			HashMap<String, HashMap<String, Integer>> attributeTrueMap, HashMap<String, HashMap<String, Integer>> attributeFalseMap) {
		HashMap<String, Double>informationGainMap = new HashMap<>();
		
		int totalCount = trueCount+falseCount;
		
		Double entropyOfTarget = -1* (getPlogBaseTwoPValue(trueCount, totalCount)+ getPlogBaseTwoPValue(falseCount, totalCount));

		
		// Entropy of each feature
		for (String m: attributeSet) {
			
			HashMap<String, Integer> biggerMap = 
			attributeFalseMap.get(m).size() > attributeTrueMap.get(m).size() ? attributeFalseMap.get(m) : attributeTrueMap.get(m);
			
			double combinedEntropyofAllValuesForVariable = 0;
			
			for (Map.Entry<String, Integer> itr: biggerMap.entrySet()) {
				int valueTrueCount = attributeTrueMap.get(m).getOrDefault(itr.getKey(), 0);
				int valueFalseCount = attributeFalseMap.get(m).getOrDefault(itr.getKey(), 0);
				int totalValCount = valueTrueCount + valueFalseCount;
				double tempVal = -1*((double)totalValCount/(double)totalCount)* (getPlogBaseTwoPValue(valueTrueCount,totalValCount)+getPlogBaseTwoPValue(valueFalseCount,totalValCount));
				combinedEntropyofAllValuesForVariable = combinedEntropyofAllValuesForVariable + tempVal;
			}
			
			informationGainMap.put(m, entropyOfTarget- combinedEntropyofAllValuesForVariable);			
		}
		
		return informationGainMap;
	}
	
	private double getPlogBaseTwoPValue(int num, int den) {
		if (num == 0) return 0;
		double val = (double)num/(double)den;
		double calculation = val* Math.log(val)/Math.log(2);
		return calculation;
	}
	
	private String getMostImportantVariable(HashMap<String, Double> informationGainMap) {
		double max = Integer.MIN_VALUE;
		String importantVar = "";
		for (Map.Entry<String, Double> m : informationGainMap.entrySet()) {
			if (m.getValue() > max) {
				max = m.getValue();
				importantVar = m.getKey();
			}
		}
		return importantVar;
	}
		
	private TreeNode getPrularityClassification(int trueCount, int falseCount) {
		if (trueCount > falseCount) {
			return new TreeNode("leaf","1");
		} else 
			return new TreeNode("leaf","0");
	}
	
	private int[] getTrueFalseCount(HashSet<Integer> elements, String targetAttribute) {
		int trueCount = 0;
		int falseCount = 0;
		
		int[] trueFalseArray = new int[2];
		
		for (int index : elements) {
			String response = trainData.get(index).get(positionIndexMap.get(targetAttribute));
			if ("1".equals(response))
				trueCount++;
			else falseCount++;
		}
		
		trueFalseArray[0] = trueCount;
		trueFalseArray[1] = falseCount;
		return trueFalseArray;
	}
	
	private String classify(TreeNode node, List<String> queryList) {
		
		if ("leaf".equals(node.name)) {
			return node.value;
		}
		
		node = node.edgeMap.get(queryList.get(positionIndexMap.get(node.name)));
		return classify(node, queryList);
	}
	
	private String getImpVarByAlternative(HashSet<String> attributeSet, HashMap<String, HashMap<String, Integer>> attributeTrueMap, HashMap<String, HashMap<String, Integer>> attributeFalseMap) {
		// Choose the variable with maximum children
		// This will allow us to narrow down the list as the dividing factor is higher
		// If the two nodes have same children, choose the node with maximum diff of False-True count
		int maxChildCount = Integer.MIN_VALUE;
		String mostImpVar = "";
		HashMap<Integer, ArrayList<String>> countMap = new HashMap<>();
		
		for (String attribute: attributeSet) {
			maxChildCount = Math.max(attributeTrueMap.get(attribute).size(), attributeFalseMap.get(attribute).size());
			
			if (!countMap.containsKey(maxChildCount)) {
				countMap.put(maxChildCount, new ArrayList<String>());
			}
			countMap.get(maxChildCount).add(attribute);			
		}
		
		int maxVal = Integer.MIN_VALUE;
		
		// Getting maximum value
		for(Map.Entry<Integer, ArrayList<String>> m: countMap.entrySet()) {
			if (m.getKey()> maxVal) {
				maxVal = m.getKey();
			}
		}
		
		// Choosing the variable
		if (countMap.get(maxVal).size()==1) {
			mostImpVar = countMap.get(maxVal).get(0);
		} else {		
			// Choose element with max true-false difference
			int maxTrueFalseDiff = Integer.MIN_VALUE;
			
			
			for (String attribute: countMap.get(maxVal)) {
				int trueCount = 0;
				int falseCount = 0;
				
				for (Map.Entry<String, Integer> m : attributeTrueMap.get(attribute).entrySet()) {
					trueCount = trueCount + m.getValue();
				}
				
				for (Map.Entry<String, Integer> m : attributeFalseMap.get(attribute).entrySet()) {
					falseCount = falseCount + m.getValue();
				}
				
				if (Math.abs(trueCount-falseCount) > maxTrueFalseDiff) {
					maxTrueFalseDiff = Math.abs(trueCount-falseCount);
					mostImpVar = attribute;
				}				
			}
			
		}
		
		return mostImpVar;
	}
}
