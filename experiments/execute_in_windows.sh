# 1) Compile in terminal
# 2) run in terminal
javac -cp h2o-genmodel.jar -J-Xms2g -J-XX:MaxPermSize=128m main.java && \
	java -cp .;h2o-genmodel.jar main
