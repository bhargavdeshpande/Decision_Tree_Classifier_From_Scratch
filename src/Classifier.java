
public class Classifier {

	public static void main(String[] args) {
		DecisionTreeClassifier dtc = new DecisionTreeClassifier();
		dtc.runAlgorithm(args[0], args[1], args[2], Integer.parseInt(args[3].trim()));
	}

}
