import os
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    Environment,
    Data,
    BuildContext,
)
from azure.ai.ml import command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.constants import AssetTypes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class AzureMLSetup:
    """Setup class for Azure ML training jobs."""
    
    def __init__(self):
        """Initialize Azure ML client and configuration."""
        # Azure ML configuration
        self.subscription_id = "60b8cee9-cee6-4922-b802-78266ea249f3"
        self.resource_group = "archivai"
        self.workspace_name = "archivai-ai"
        self.job_name = None
        # Initialize Azure ML client
        self.credential = DefaultAzureCredential()
        self.ml_client = MLClient(
            credential=self.credential,
            subscription_id=self.subscription_id,
            resource_group_name=self.resource_group,
            workspace_name=self.workspace_name
        )
        
        print(f"Azure ML Client initialized for workspace: {self.workspace_name}")
    
    def create_custom_environment(self, 
                                environment_name: str = "jina-training-env",
                                environment_version: str = "1.0"):
        """
        Create a custom environment for training.
        
        Args:
            environment_name (str): Name of the environment
            environment_version (str): Version of the environment
            
        Returns:
            Environment: Created environment object
        """
        # Create environment from Docker image
        environment = Environment(
            name=environment_name,
            version=environment_version,
            build=BuildContext(path="docker-context"),
            description="Custom environment for Jina AI training with transformers and MLflow"
        )
        
        try:
            created_env = self.ml_client.environments.create_or_update(environment)
            print(f"Environment '{environment_name}:{environment_version}' created successfully")
            return created_env
        except Exception as e:
            print(f"Error creating environment: {e}")
            # Try to get existing environment
            try:
                existing_env = self.ml_client.environments.get(environment_name, environment_version)
                print(f"Using existing environment: {environment_name}:{environment_version}")
                return existing_env
            except:
                print(f"Failed to create or retrieve environment")
                return None
    
    def upload_data(self, 
                   data_path: str,
                   data_name: str = "jina-training-data",
                   data_version: str = "1.0"):
        """
        Upload training data to Azure ML.
        
        Args:
            data_path (str): Local path to the data file
            data_name (str): Name for the data asset
            data_version (str): Version of the data asset
            
        Returns:
            Data: Created data asset
        """
        data_asset = Data(
            name=data_name,
            version=data_version,
            description="Training data for Jina AI classification",
            path=data_path,
            type=AssetTypes.URI_FILE
        )
        
        try:
            created_data = self.ml_client.data.create_or_update(data_asset)
            print(f"Data asset '{data_name}:{data_version}' uploaded successfully")
            return created_data
        except Exception as e:
            print(f"Error uploading data: {e}")
            # Try to get existing data
            try:
                existing_data = self.ml_client.data.get(data_name, data_version)
                print(f"Using existing data asset: {data_name}:{data_version}")
                return existing_data
            except:
                print(f"Failed to upload or retrieve data asset")
                return None
    
    def create_training_job(self,
                            data_path: str,
                            experiment_name: str = "jina-classification-training",
                            compute_target: str = "gpu-cluster",
                            environment_name: str = "jina-training-env",
                            environment_version: str = "1.0",
                            data_name: str = "jina-training-data",
                            data_version: str = "1.0",
                            run_name: str = None,
                            ):
        """
        Create and submit a training job to Azure ML.
        
        Args:
            experiment_name (str): Name of the experiment
            compute_target (str): Name of the compute target
            environment_name (str): Name of the environment
            environment_version (str): Version of the environment
            data_name (str): Name of the data asset
            data_version (str): Version of the data asset
            run_name (str, optional): Name for the specific run
            
        Returns:
            Command: Submitted job object
        """
        if run_name is None:
            import datetime
            run_name = f"jina-training-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        # Create the training command
        command_job = command(
            experiment_name=experiment_name,
            display_name=run_name,
            description="Fine-tuning Jina AI model for document classification",
            # Command to run
            command=f"python train_script.py --data_path ${{inputs.training_data}} --run_name {run_name}",
            
            # Code location
            code="./",  # Current directory containing all training files
            
            # Environment
            environment=f"{environment_name}:{environment_version}",
            
            # Compute target
            compute=compute_target,
            
            # Inputs
            inputs={
                "training_data": Input(type="uri_file", path=data_path),
                # "run_name": Input(type="string", value="test_run_1")
            },
            
            # Tags
            tags={
                "model": "jina-ai",
                "task": "classification",
                "framework": "transformers"
            }
        )
        
        try:
            submitted_job = self.ml_client.jobs.create_or_update(command_job)
            self.job_name = submitted_job.name
            print(f"Training job submitted successfully!")
            print(f"Job name: {submitted_job.name}")
            print(f"Job status: {submitted_job.status}")
            print(f"Job URL: {submitted_job.studio_url}")
            return submitted_job
        except Exception as e:
            print(f"Error submitting training job: {e}")
            return None
    
    def get_run_id_from_job(self, job_name: str = None):
        if job_name is None:
            if self.job_name is None:
                raise ValueError("Job name must be provided or set during job submission.")
            job_name = self.job_name
        job = self.ml_client.jobs.get(job_name)
        output_dir = job.outputs["output"].path
        run_id_file = os.path.join(output_dir, "run_id.txt")
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
        return run_id
    
    def monitor_job(self, job_name: str):
        """
        Monitor the status of a training job.
        
        Args:
            job_name (str): Name of the job to monitor
        """
        try:
            job = self.ml_client.jobs.get(job_name)
            print(f"Job '{job_name}' status: {job.status}")
            if job.studio_url:
                print(f"Monitor at: {job.studio_url}")
            return job
        except Exception as e:
            print(f"Error monitoring job: {e}")
            return None
    
    def list_compute_targets(self):
        """List available compute targets."""
        try:
            compute_targets = self.ml_client.compute.list()
            print("Available compute targets:")
            for compute in compute_targets:
                print(f"  - {compute.name} (type: {compute.type}, state: {compute.provisioning_state})")
        except Exception as e:
            print(f"Error listing compute targets: {e}")
    
    def get_job_logs(self, job_name: str):
        """
        Get logs from a training job.
        
        Args:
            job_name (str): Name of the job
        """
        try:
            # This would typically require additional setup for log streaming
            job = self.ml_client.jobs.get(job_name)
            print(f"Job logs can be viewed at: {job.studio_url}")
            return job
        except Exception as e:
            print(f"Error getting job logs: {e}")
            return None


def main():
    # """Main function for setting up and running Azure ML training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup Azure ML Training Job")
    parser.add_argument("--action", type=str, choices=["setup", "submit", "monitor", "list_compute"], 
                    default="setup", help="Action to perform")
    parser.add_argument("--data_path", type=str, default="extracted_data.csv", 
                    help="Path to training data")
    parser.add_argument("--compute_target", type=str, default="H100-Cluster", 
                    help="Compute target name")
    parser.add_argument("--experiment_name", type=str, default="jina-classification-training",
                    help="Experiment name")
    parser.add_argument("--run_name", type=str, help="Run name")
    parser.add_argument("--job_name", type=str, help="Job name to monitor", default="jina-training-job")
    path = "Draft/train_app/extracted_data.csv"
    args = parser.parse_args()
    
    # # Initialize Azure ML setup
    azure_setup = AzureMLSetup()

    if args.action == "list_compute":
        azure_setup.list_compute_targets()

    # elif args.action == "setup":


    elif args.action == "submit":
        print("=== Setting up Azure ML Environment ===")


        env_name = "jina-training-ai-env"
        env_version = "2.0"
        # Create custom environment
        # env = azure_setup.create_custom_environment(env_name, env_version)
        # if env is None:
        #     print("Failed to create environment")
        #     return

        # Upload data
        data_asset = azure_setup.upload_data(args.data_path)
        if data_asset is None:
            print("Failed to upload data")
            return

        print("Setup completed successfully!")
        print("==============================")
        print("=== Submitting Training Job ===")

        job = azure_setup.create_training_job(
            environment_name=env_name,
            environment_version=env_version,
            data_path=data_asset.path,
            experiment_name=args.experiment_name,
            compute_target=args.compute_target,
            run_name=args.run_name,

        )

        if job:
            print(f"Job submitted: {job.name}")
        else:
            print("Failed to submit job")

    elif args.action == "monitor":
        if not args.job_name:
            print("Please provide --job_name to monitor")
            return

        print(f"=== Monitoring Job: {args.job_name} ===")
        azure_setup.monitor_job(args.job_name)


if __name__ == "__main__":
    main()