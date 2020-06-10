import java.io.IOException;

import cli.AccuracyTest;


public class Main {

	public static void main(String[] args) throws IOException, InterruptedException {
		AccuracyTest at = new AccuracyTest("EXP");
		at.run();
	}

}
