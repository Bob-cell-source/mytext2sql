import argparse
import json
import os
from dataflow import get_logger
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download

from dataflow.operators.text2sql import (
    SQLVariationGenerator,
    Text2SQLQuestionGenerator,
    Text2SQLPromptGenerator,
    Text2SQLCoTGenerator,
    Text2SQLCoTVotingGenerator
)
from dataflow.operators.text2sql import (
    SQLExecutabilityFilter,
    Text2SQLCorrespondenceFilter
)
from dataflow.operators.text2sql import (
    SQLComponentClassifier,
    SQLExecutionClassifier
)
from dataflow.prompts.text2sql import (
    Text2SQLCorrespondenceFilterPrompt,
    Text2SQLCotGeneratorPrompt,
    Text2SQLQuestionGeneratorPrompt,
    SQLVariationGeneratorPrompt,
    Text2SQLPromptGeneratorPrompt
)
from dataflow.utils.storage import FileStorage
from dataflow.serving import APILLMServing_request
from dataflow.utils.text2sql.database_manager import DatabaseManager


def download_and_extract_database(logger):
    dataset_repo_id = "Open-Dataflow/dataflow-Text2SQL-database-example"
    local_dir = "./hf_cache"
    extract_to = "./downloaded_databases"
    
    logger.info(f"Downloading and extracting database from {dataset_repo_id}...")
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    db_exp_folder_path = os.path.join(extract_to, "databases")

    if os.path.exists(db_exp_folder_path):
        return db_exp_folder_path
    
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(extract_to, exist_ok=True)
    
    downloaded_path = snapshot_download(
        repo_id=dataset_repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        resume_download=True
    )
    
    logger.info(f"Files downloaded to: {downloaded_path}")
    
    zip_path = os.path.join(downloaded_path, "databases.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Database files extracted to {extract_to}")
        return db_exp_folder_path
    else:
        raise FileNotFoundError(f"Database zip file not found at {zip_path}")

class Text2SQLRefine_APIPipeline():
    def __init__(
        self,
        db_type="mysql",
        db_root_path="",
        seed_file="../example_data/Text2SQLPipeline/tgac_pipeline_refine.jsonl",
        cache_path="./cache_tgac_refine",
        llm_api_url=None,
        llm_model_name=None,
        embedding_api_url=None,
        embedding_model_name=None,
        llm_max_workers=16,
        embedding_max_workers=16,
        mysql_config=None,
    ):
        self.logger = get_logger()
        self.db_type = db_type
        self.db_root_path = db_root_path
        self.seed_file = seed_file
        self.cache_path = cache_path
        self.llm_api_url = llm_api_url or os.getenv("DATAFLOW_LLM_API_URL", "https://api.openai.com/v1/chat/completions")
        self.llm_model_name = llm_model_name or os.getenv("DATAFLOW_LLM_MODEL", "gpt-4o")
        self.embedding_api_url = embedding_api_url or os.getenv(
            "DATAFLOW_EMBEDDING_API_URL", "https://api.openai.com/v1/embeddings"
        )
        self.embedding_model_name = embedding_model_name or os.getenv(
            "DATAFLOW_EMBEDDING_MODEL", "text-embedding-3-small"
        )
        self.llm_max_workers = int(llm_max_workers)
        self.embedding_max_workers = int(embedding_max_workers)
        self.mysql_config = mysql_config or {
            "host": os.getenv("DB_HOST", "127.0.0.1"),
            "user": os.getenv("DB_USER", "root"),
            "password": os.getenv("DB_PASSWORD", ""),
            "database": os.getenv("DB_NAME", "TGAC"),
            "port": int(os.getenv("DB_PORT", "3306")),
        }

        if not os.path.exists(self.seed_file):
            raise FileNotFoundError(f"Seed file does not exist: {self.seed_file}")

        if self.db_type == "sqlite":
            if not db_root_path:
                try:
                    self.db_root_path = download_and_extract_database(self.logger)
                    self.logger.info(f"Using automatically downloaded database at: {self.db_root_path}")
                except Exception as e:
                    self.logger.error(f"Failed to auto-download database: {e}")
                    raise
            else:
                self.logger.info(f"Using manually specified database path: {self.db_root_path}")

            if not os.path.exists(self.db_root_path):
                raise FileNotFoundError(f"Database path does not exist: {self.db_root_path}")
        elif self.db_type == "mysql":
            self.logger.info(f"Using MySQL-compatible database: {self.mysql_config['host']}:{self.mysql_config['port']}/{self.mysql_config['database']}")
        else:
            raise ValueError(f"Unsupported db_type: {self.db_type}")

        self.llm_serving = APILLMServing_request(
            api_url=self.llm_api_url,
            model_name=self.llm_model_name,
            max_workers=self.llm_max_workers
        )

        self.embedding_serving = APILLMServing_request(
            api_url=self.embedding_api_url,
            model_name=self.embedding_model_name,
            max_workers=self.embedding_max_workers
        )

        # SQLite and MySQL are currently supported
        # db_type can be sqlite or mysql, which must match your database type
        # If sqlite is selected, root_path must be provided, this path must exist and contain database files
        # If mysql is selected, host, user, password must be provided, these credentials must be correct and have access permissions
        # MySQL example:
        # database_manager = DatabaseManager(
        #     db_type="mysql",
        #     config={
        #         "host": "localhost",
        #         "user": "root",
        #         "password": "your_password",
        #         "database": "your_database_name"
        #     }
        # )
        # SQLite example:
        if self.db_type == "sqlite":
            database_manager = DatabaseManager(
                db_type="sqlite",
                config={
                    "root_path": self.db_root_path
                }
            )
        else:
            database_manager = DatabaseManager(
                db_type="mysql",
                config=self.mysql_config
            )
        self.database_manager = database_manager

        normalized_seed_file = self._normalize_seed_db_ids(self.seed_file)
        self.storage = FileStorage(
            first_entry_file_name=normalized_seed_file,
            cache_path=self.cache_path,
            file_name_prefix="dataflow_cache_step",
            cache_type="jsonl"
        )
        
        self.sql_executability_filter_step1 = SQLExecutabilityFilter(
            database_manager=database_manager
        )

        self.sql_variation_generator_step2 = SQLVariationGenerator(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            num_variations=3, # Number of variations to generate for each SQL
            prompt_template=SQLVariationGeneratorPrompt()
        )

        self.sql_executability_filter_step3 = SQLExecutabilityFilter(
            database_manager=database_manager
        )

        self.text2sql_question_generator_step4 = Text2SQLQuestionGenerator(
            llm_serving=self.llm_serving,
            embedding_serving=self.embedding_serving,
            database_manager=database_manager,
            question_candidates_num=3,
            prompt_template=Text2SQLQuestionGeneratorPrompt()
        )

        self.text2sql_correspondence_filter_step5 = Text2SQLCorrespondenceFilter(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            prompt_template=Text2SQLCorrespondenceFilterPrompt()
        )

        self.text2sql_prompt_generator_step6 = Text2SQLPromptGenerator(
            database_manager=database_manager,
            prompt_template=Text2SQLPromptGeneratorPrompt()
        )

        self.sql_cot_generator_step7 = Text2SQLCoTGenerator(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            prompt_template=Text2SQLCotGeneratorPrompt()
        )

        self.sql_cot_voting_generator_step8 = Text2SQLCoTVotingGenerator(
            database_manager=database_manager
        )

        self.sql_component_classifier_step9 = SQLComponentClassifier(
            difficulty_thresholds=[2, 4, 6],
            difficulty_labels=['easy', 'medium', 'hard', 'extra']
        )

        self.sql_execution_classifier_step10 = SQLExecutionClassifier(
            llm_serving=self.llm_serving,
            database_manager=database_manager,
            num_generations=10,
            difficulty_thresholds=[2, 5, 9],
            difficulty_labels=['extra', 'hard', 'medium', 'easy']
        )

    def _normalize_seed_db_ids(self, seed_file: str) -> str:
        os.makedirs(self.cache_path, exist_ok=True)
        available_databases = self.database_manager.list_databases()
        lower_name_map = {name.lower(): name for name in available_databases}

        changed_rows = 0
        unknown_db_ids = set()
        normalized_rows = []

        with open(seed_file, "r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)
                db_id = row.get("db_id")
                if not db_id:
                    raise ValueError(f"Missing db_id in seed file at line {line_number}")

                if db_id in lower_name_map.values():
                    normalized_rows.append(row)
                    continue

                normalized_db_id = lower_name_map.get(str(db_id).lower())
                if normalized_db_id:
                    row["db_id"] = normalized_db_id
                    changed_rows += 1
                else:
                    unknown_db_ids.add(str(db_id))

                normalized_rows.append(row)

        if unknown_db_ids:
            self.logger.warning(
                f"Some seed db_id values are not present in the discovered database registry: {sorted(unknown_db_ids)}"
            )

        if changed_rows == 0:
            return seed_file

        normalized_seed_path = os.path.join(self.cache_path, "_normalized_seed.jsonl")
        with open(normalized_seed_path, "w", encoding="utf-8") as f:
            for row in normalized_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        self.logger.info(
            f"Normalized db_id casing for {changed_rows} seed rows. Using normalized seed file: {normalized_seed_path}"
        )
        return normalized_seed_path

    def _assert_nonempty_step_output(self, step_file: str, step_name: str) -> None:
        if not os.path.exists(step_file):
            raise RuntimeError(f"{step_name} did not produce an output file: {step_file}")

        if os.path.getsize(step_file) == 0:
            raise RuntimeError(
                f"{step_name} produced an empty result file. "
                "This usually means all rows were filtered out before the next step. "
                "Please check db_id/database name matching and whether EXPLAIN is valid for the seed SQLs."
            )

    def forward(self):

        sql_key = "SQL"
        db_id_key = "db_id"
        question_key = "question"
        evidence_key = "evidence"

        self.sql_executability_filter_step1.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key
        )
        self._assert_nonempty_step_output(
            os.path.join(self.cache_path, "dataflow_cache_step_step1.jsonl"),
            "sql_executability_filter_step1"
        )

        self.sql_variation_generator_step2.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            output_sql_variation_type_key="sql_variation_type"
        )

        self.sql_executability_filter_step3.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key
        )

        self.text2sql_question_generator_step4.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            output_question_key=question_key,
            output_evidence_key=evidence_key
        )

        self.text2sql_correspondence_filter_step5.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            input_question_key=question_key,
            input_evidence_key=evidence_key
        )

        self.text2sql_prompt_generator_step6.run(
            storage=self.storage.step(),
            input_question_key=question_key,
            input_db_id_key=db_id_key,
            input_evidence_key=evidence_key,
            output_prompt_key="prompt"
        )

        self.sql_cot_generator_step7.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_question_key=question_key,
            input_db_id_key=db_id_key,
            input_evidence_key=evidence_key,
            output_cot_key="cot_reasoning"
        )

        self.sql_cot_voting_generator_step8.run(
            storage=self.storage.step(),
            input_cot_responses_key="cot_responses",
            input_db_id_key=db_id_key,
            output_cot_key="cot_reasoning"
        )

        self.sql_component_classifier_step9.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            output_difficulty_key="sql_component_difficulty"
        )

        self.sql_execution_classifier_step10.run(
            storage=self.storage.step(),
            input_sql_key=sql_key,
            input_db_id_key=db_id_key,
            input_prompt_key="prompt",
            output_difficulty_key="sql_execution_difficulty"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dataflow Text2SQL refine pipeline for TGAC-style seeds.")
    parser.add_argument("--db-type", default=os.getenv("DATAFLOW_DB_TYPE", "mysql"), choices=["mysql", "sqlite"])
    parser.add_argument(
        "--db-root-path",
        default=os.getenv("DATAFLOW_DB_ROOT_PATH", ""),
        help="SQLite root path. Only used when --db-type=sqlite.",
    )
    parser.add_argument(
        "--seed-file",
        default=os.getenv(
            "DATAFLOW_SEED_FILE",
            "../example_data/Text2SQLPipeline/tgac_pipeline_refine.jsonl",
        ),
        help="Input jsonl seed file.",
    )
    parser.add_argument(
        "--cache-path",
        default=os.getenv("DATAFLOW_CACHE_PATH", "./cache_tgac_refine"),
        help="Cache directory for intermediate Dataflow outputs.",
    )
    parser.add_argument(
        "--llm-api-url",
        default=os.getenv("DATAFLOW_LLM_API_URL", "https://api.openai.com/v1/chat/completions"),
    )
    parser.add_argument(
        "--llm-model-name",
        default=os.getenv("DATAFLOW_LLM_MODEL", "gpt-4o"),
    )
    parser.add_argument(
        "--embedding-api-url",
        default=os.getenv("DATAFLOW_EMBEDDING_API_URL", "https://api.openai.com/v1/embeddings"),
    )
    parser.add_argument(
        "--embedding-model-name",
        default=os.getenv("DATAFLOW_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    parser.add_argument(
        "--llm-max-workers",
        type=int,
        default=int(os.getenv("DATAFLOW_LLM_MAX_WORKERS", "16")),
    )
    parser.add_argument(
        "--embedding-max-workers",
        type=int,
        default=int(os.getenv("DATAFLOW_EMBEDDING_MAX_WORKERS", "16")),
    )
    args = parser.parse_args()

    model = Text2SQLRefine_APIPipeline(
        db_type=args.db_type,
        db_root_path=args.db_root_path,
        seed_file=args.seed_file,
        cache_path=args.cache_path,
        llm_api_url=args.llm_api_url,
        llm_model_name=args.llm_model_name,
        embedding_api_url=args.embedding_api_url,
        embedding_model_name=args.embedding_model_name,
        llm_max_workers=args.llm_max_workers,
        embedding_max_workers=args.embedding_max_workers,
    )
    model.forward()
