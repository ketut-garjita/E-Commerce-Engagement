## Train Model

**Notes:**

- If you want to train the model using Pandas locally, follow Steps 1 and 3 (choose **Option 1: Pandas**).

- If you want to train the model using PySpark, follow Steps 1, 2, 3 (choose **Option 2: PySpark**).

_PS: Training the model will take approximately 1 hour on a Linux Virtual Machine (VM) with 16GB of memory and 8 CPUs._

   
#### **STEP 1: Setup Kaggle Account Authorization on Local Linux Machine**

   - Create a Kaggle Account
  
     Visit [Kaggle](https://www.kaggle.com/) and sign up if you don’t already have an account.

   - Generate API Credentials
        - Log in to Kaggle.
        - Go to your account settings:
        - Click on your profile picture (top right corner) → **Account**.
        - Scroll down to the **API** section and click **Create New API Token**. This will download a file called **kaggle.json** to your computer.
        -  Move the API Token to the Correct Location

   -  Place the kaggle.json file in the appropriate directory:  
      ```
      mkdir -p ~/.kaggle
      mv /path/to/kaggle.json ~/.kaggle/
      chmod 600 ~/.kaggle/kaggle.json       
      ```

   - Install Kaggle Python Package
  
      Ensure you have the kaggle package installed. Run:
  
      ```
      pip install kaggle 
      ```
      
   - Verify the Setup
    
      Test the setup by running :
  
      ```
      kaggle datasets list   
      ```
      
      This command should display a list of datasets without errors.
  
      
#### **STEP 2: Install and Setup Hadoop on Linux Machine**

   *==> NOTE: This is for option of train model with pyspark tool (train_model_pyspark.py) !!!*
    
   - Install Java

     Hadoop requires Java. Install the latest version of OpenJDK:
   
     ```
     sudo apt update
     sudo apt install -y openjdk-21-jdk     
     ```
    
     Verify Java installation:
   
     ```
     java -version  
     ```
    
   - Download Hadoop

     Visit the Apache Hadoop downloads page and copy the latest stable version link.

     [https://downloads.apache.org/hadoop/common/](https://downloads.apache.org/hadoop/common/)

     Download it using wget:
   
     ```
     wget https://downloads.apache.org/hadoop/common/hadoop-<version>/hadoop-<version>.tar.gz     
     ```
    
     Extract the archive:
   
     ```
     tar -xzf hadoop-<version>.tar.gz
     sudo mv hadoop-<version> /usr/local/hadoop      
     ```
    
   - Setup Environment Variables

     Add Hadoop and Java paths to your environment variables. Edit the .bashrc file:
   
     ```
     nano ~/.bashrc      
     ```

     Add the following lines at the end:

     ```
     export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
     export HADOOP_HOME=/usr/local/hadoop
     export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
     export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
     export HADOOP_MAPRED_HOME=$HADOOP_HOME
     export HADOOP_COMMON_HOME=$HADOOP_HOME
     export HADOOP_HDFS_HOME=$HADOOP_HOME
     export HADOOP_YARN_HOME=$HADOOP_HOME      
     ```
    
     Apply the changes:
   
     ```
     source ~/.bashrc      
     ```
    
   - Configure Hadoop

     Modify configuration files in the HADOOP_HOME/etc/hadoop directory.

     **hadoop-env.sh**

     Set Java path:
     ```
     export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64     
     ```
    
     **core-site.xml**

     Configure Hadoop's default file system:

     ```
      <configuration>
          <property>
              <name>fs.defaultFS</name>
              <value>hdfs://localhost:9000</value>
          </property>
      </configuration>      
     ```
    
     **hdfs-site.xml**

     Configure directories for NameNode and DataNode:

     ```
      <configuration>
          <property>
              <name>dfs.replication</name>
              <value>1</value>
          </property>
          <property>
              <name>dfs.namenode.name.dir</name>
              <value>file:///usr/local/hadoop/hdfs/namenode</value>
          </property>
          <property>
              <name>dfs.datanode.data.dir</name>
              <value>file:///usr/local/hadoop/hdfs/datanode</value>
          </property>
      </configuration>      
     ```
        
   - Format the NameNode

     Before starting Hadoop services, format the NameNode:

     ```
      hdfs namenode -format      
     ```
    
   - Start Hadoop Services

     ```
      start-dfs.sh            
     ```
    
   - Verify Services

     Access the following web interfaces:

     NameNode UI: [http://localhost:9870](http://localhost:9870)

     Check running services:
   
     ```
      jps      
     ```
    
     You should see NameNode, DataNode, SecondaryNameNode, ResourceManager.

  - Test HDFS

      Create a directory in HDFS:
   
       ```
        hdfs dfs -mkdir /user
        hdfs dfs -mkdir /user/yourusername
        hdfs dfs -ls /        
       ```
   
#### **STEP 3: Train and Save Model**

   Run the model training script and save the model in TensorFlow SavedModel format.

   Install Package Dependencies

   ```
   cd E-Commerce-Engagement
   pip install -r requirements.txt  
   ```

   **OPTION 1 : Using Pandas Tool and Dataset on Local File System**

   ```
   cd E-Commerce-Engagement/src
   python train_model_pandas.py  
   ```

   **OPTION 2 : Using Pyspark Tool and Dataset on Hadoop Distributed File System (HDFS)**

   ```
   pip install pyspark
    
   cd E-Commerce-Engagement/src
   python train_model_pyspark.py   
   ```
