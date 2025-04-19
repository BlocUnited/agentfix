# ve_concept_creation.py

import autogen
import copy
from pydantic import BaseModel, ValidationError
from pydantic.json_schema import model_json_schema
from autogen import Agent, ConversableAgent, AssistantAgent, OpenAIWrapper, UserProxyAgent, gather_usage_summary
from autogen.io.websockets import IOWebsockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from threading import Thread
from queue import Queue
from bson import ObjectId  # Import ObjectId to use in your query
import asyncio
import uuid
import uvicorn
from pymongo import MongoClient, DESCENDING
import os
import json
import logging
import traceback
from motor.motor_asyncio import AsyncIOMotorClient
from shared_app import app, add_initialization_coroutine, shared_websocket_manager
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple, Union
try:
    from termcolor import colored
except ImportError:
    def colored(x, *args, **kwargs):
        return x

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
app = FastAPI()

# MongoDB connection (using Motor for async operations)
connection_string = "mongodb+srv://dev:OQaCVOXwkwGyaBJu@mozaiks.d82loyy.mongodb.net/"
mongo_client = AsyncIOMotorClient(connection_string)

# Connect to your Mozaiks database
db1 = mongo_client['MozaiksDB']  # Database name
enterprises_collection = db1['Enterprises']  # Collection name.

# Connect to your Agents database
db2 = mongo_client['autogen_ai_agents']  # Database name
concepts_collection = db2['Concepts']  # Collection name
llm_config_collection = db2['LLMConfig']  # Collection name

# Global variables
client = None
llm_config = None
llm_config_vision = None
llm_config_concept = None
ConceptVerificationConvo = None
enterprise_id = None
user_id = None

# Define structured models for features and third-party integrations
class SuggestedFeature(BaseModel):
    feature_title: str
    description: str

class ThirdPartyIntegration(BaseModel):
    technology_title: str
    description: str

# Update VisionResponse to reflect the structured data
class VisionResponse(BaseModel):
    core_focus: str
    monetization: int
    suggested_features: List[SuggestedFeature]  # Updated to list of objects
    third_party_integrations: List[ThirdPartyIntegration]  # Updated to list of objects
    liked_apps: List[str]

# Ensure that VisionAgentOutputs correctly stores multiple VisionResponse objects
class VisionAgentOutputs(BaseModel):
    output_response: List[VisionResponse]

class FeatureOutput(BaseModel):
    feature_title: str
    description: str

class ConceptRefinementResponse(BaseModel):
    tagline: str
    narrative: str
    features: List[FeatureOutput]
    important_note: str
    status: int  # 0 for continuing, 1 for ending the chat

class ConceptRefinementOutputs(BaseModel):
    output_response: List[ConceptRefinementResponse]

# Modify your load_config function
async def load_config():
    llm_config_doc = await llm_config_collection.find_one()
    if not llm_config_doc:
        raise ValueError("No LLM configuration found in the database")

    openai_ApiKey = llm_config_doc.get('ApiKey')
    model_name = llm_config_doc.get('Model', 'o3-mini')

    if not openai_ApiKey:
        raise ValueError("OpenAI API key not found in the LLM configuration")

    model_pricing = {
        "o3-mini": [0.0011, 0.0044]
    }

    price = model_pricing.get(model_name)
    if not price:
        raise ValueError(f"Pricing information for model '{model_name}' is not available")

    config_list = [{
        "model": model_name,
        "api_key": openai_ApiKey,
        "price": price
    }]

    client = OpenAIWrapper(config_list=config_list)
    return client, config_list

async def cc_initialize():
    global client, llm_config, llm_config_vision, llm_config_concept

    try:
        client, config_list = await load_config()
        client, config_list_vision = await load_config()
        client, config_list_concept = await load_config()        
    except ValueError as e:
        logger.error(f"Error in load_config: {e}")
        return

    # Add response formats to config_list
    for config in config_list_vision:
        config["response_format"] = VisionAgentOutputs

    # Add response formats to config_list
    for config in config_list_concept:
        config["response_format"] = ConceptRefinementOutputs

    llm_config = {
        "timeout": 600,
        "cache_seed": 163,
        "config_list": config_list
    }

    llm_config_vision = {
        "timeout": 600,
        "cache_seed": 163,
        "config_list": config_list_vision
    }

    llm_config_concept = {
        "timeout": 600,
        "cache_seed": 163,
        "config_list": config_list_concept
    }

    logger.info("CC initialization completed.")
 
async def load_concept_verification(enterprise_id=None):
    try:
        # Create query based on whether enterprise_id is provided
        query = {}
        if enterprise_id:
            enterprise_id_str = str(enterprise_id)
            query["enterprise_id"] = enterprise_id_str
            
        # Load latest concept verification conversation
        latest_concept = await concepts_collection.find_one(
            query,
            sort=[("ConceptCode", DESCENDING)]
        )
        
        if latest_concept:
            ConceptVerificationConvo = latest_concept.get('ConceptVerificationConvo', [])
            logger.info(f"Loaded concept verification conversation for concept code: {latest_concept.get('ConceptCode')}")
        else:
            logger.warning("No existing concept verification analysis found. Using empty list.")
            ConceptVerificationConvo = []

        return ConceptVerificationConvo

    except Exception as e:
        logger.error(f"Error in load_concept_verification: {str(e)}")
        logger.error(traceback.format_exc())
        raise

class AsyncConversableWrapper(ConversableAgent):
    async def a_respond(self, recipient, messages, sender, config, last_message=None):
        logger.info(f"a_respond called with last_message: {last_message}")

        message_content = None
        should_append = True

        # Try to extract status from last_message with improved error handling
        if last_message:
            try:
                if isinstance(last_message, str):
                    # Only attempt to parse as JSON if it looks like JSON
                    if last_message.strip().startswith('{') and last_message.strip().endswith('}'):
                        try:
                            message_content = json.loads(last_message)
                        except json.JSONDecodeError:
                            message_content = last_message  # Use as plain text if parsing fails
                    else:
                        message_content = last_message  # Not JSON-formatted, use as is
                else:
                    message_content = last_message  # Already an object, not a string

                # Check status only if we have a dictionary
                if isinstance(message_content, dict) and message_content.get('status') == 1:
                    should_append = False
                    logger.info("Skipping append due to status = 1")
            except Exception as e:
                logger.warning(f"Error processing last_message: {str(e)}")
                message_content = last_message  # Fallback to original message

        # Ensure each message has a role and name
        for message in messages:
            if "role" not in message:
                message["role"] = "user" if sender == "user_proxy" else "assistant"
            if "name" not in message or not message["name"].strip():
                message["name"] = "unknown"

        # Agent-specific behavior overrides
        if self.name == "Data_Agent" and isinstance(message_content, dict) and message_content.get('status') == 1:
            should_append = False
            logger.info(f"Skipping append due to status = 1 from {self.name}")
        elif self.name == "Concept_Refinement_Agent":
            should_append = True

        # Conditionally append last message
        if last_message and last_message not in messages and should_append:
            logger.info(f"Appending last_message: {last_message}")
            messages.append({"role": "assistant", "content": last_message, "name": self.name})
        else:
            logger.info(f"Skipping appending last_message: {last_message}")

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, self.generate_reply, messages, sender)
        return True, response

class UserProxyWebAgent(UserProxyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iostream: Optional[IOWebsockets] = None
        self.message_queue: Optional[asyncio.Queue] = None
        self.groupchat_manager: Optional[AsyncGroupChatManager] = None
        self.input_event = asyncio.Event()
        self.last_input = None

    async def a_initiate_chat(self, recipient, message):
        logger.info(f"a_initiate_chat called by {self.__class__.__name__} with recipient: {recipient.__class__.__name__}")
        if isinstance(recipient, AsyncGroupChatManager):
            await recipient.a_receive(message, self)
        else:
            logger.info(f"Unexpected recipient type: {type(recipient)}")

    async def a_get_human_input(self, prompt: str) -> str:
        logger.info(f"a_get_human_input called with prompt: {prompt}")
        if self.iostream and not self.iostream.websocket.client_state.DISCONNECTED:
            try:
                self.input_event.clear()
                await self.input_event.wait()
                return self.last_input
            except Exception as e:
                logger.info(f"Unable to get human input: {str(e)}")
                return ""
        else:
            logger.info("WebSocket not connected. Skipping human input.")
            return ""

    async def receive_input(self, input_data: str):
        try:
            # Try to parse as JSON
            reply_json = json.loads(input_data)
            content = reply_json.get("content", "")
            logger.info(f"Parsed JSON input: {content}")
        except json.JSONDecodeError:
            # If not JSON, use the raw input
            content = input_data
            logger.info(f"Using raw input: {content}")

        self.last_input = content
        self.input_event.set()

    async def a_respond(self, recipient, messages, sender, config, last_message=None):
        logger.info(f"a_respond called by {self.__class__.__name__} with sender: {sender.name if hasattr(sender, 'name') else 'Unknown'} and recipient: {recipient.__class__.__name__ if recipient else 'NoneType'}")

        if hasattr(sender, 'name') and sender.name == "Feedback_Agent":
            logger.info(f"user_proxy awaiting user input after Feedback_Agent's message.")
            
            try:
                reply = await self.a_get_human_input("")
                if not reply:
                    logger.info("No human input received. Skipping user_proxy response.")
                    return False, None  # Skip the response and don't trigger the next agent

                if reply.lower() == "exit":
                    logger.info("User chose to exit. Ending session.")
                    return True, None  # End the chat session if 'exit' is typed

                logger.info(f"User feedback received: {reply}")
                return True, reply
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected while waiting for user input.")
                return False, None  # Don't end the chat session, just skip this response

        logger.info(f"Sender {sender.name if hasattr(sender, 'name') else 'Unknown'} is not Feedback_Agent, proceeding with default behavior.")
        return False, None

class WorkflowManager:
    def __init__(self, llm_config, websocket: WebSocket, chat_id: str, user_id: str, enterprise_id: str, client):
        self.llm_config = llm_config  # Single unified config
        self.websocket = websocket
        self.chat_id = chat_id
        self.user_id = user_id        
        self.enterprise_id = enterprise_id
        self.client = client
        self.AgentHistory = []
        self.message_queue = asyncio.Queue()
        self.agents = self.create_agents()
        self.groupchat = self.create_groupchat()
        self.groupchat_manager = self.create_groupchat_manager()
        self.autogengroupchatmanager = autogen.GroupChatManager(groupchat=self.groupchat, llm_config=llm_config) 
        self.IterationCount = 0
        self.Concept_Refinement_Agent_count = 0
        self.UserFeedbackCount = 0
        self.agent_dict = {agent.name: agent for agent in self.agents}
        self.LastSpeaker = None
        self.SessionID = str(uuid.uuid4())  # Generate a unique session ID
        self.cumulative_PromptTokens = 0
        self.cumulative_CompletionTokens = 0
        self.cumulative_TotalTokens = 0
        self.cumulative_TotalCost = 0.0

    def initialize_new_session_for_tracking(self):
        self.SessionID = str(uuid.uuid4())
        self.cumulative_PromptTokens = 0
        self.cumulative_CompletionTokens = 0
        self.cumulative_TotalTokens = 0
        self.cumulative_TotalCost = 0.0
        logger.info(f"Initialized a new tracking session with SessionID: {self.SessionID}")

    def initialize_new_chat(self):
        self.AgentHistory = []
        self.IterationCount = 0
        self.Concept_Refinement_Agent_count = 0
        self.UserFeedbackCount = 0
        self.LastSpeaker = None
        self.initialize_new_session_for_tracking()
        logger.info(f"Initialized a new chat and tracking session")

    async def update_CreationChatStatus(self, status):
        try:
            # Ensure enterprise_id is string for concepts_collection
            enterprise_id_str = str(self.enterprise_id)
            
            latest_concept = await concepts_collection.find_one(
                {"enterprise_id": enterprise_id_str},
                sort=[("ConceptCode", DESCENDING)]
            )
            
            if latest_concept:
                current_status = latest_concept.get('CreationChatStatus')
                if current_status != status:
                    update_result = await concepts_collection.update_one(
                        {'_id': latest_concept['_id']},
                        {'$set': {'CreationChatStatus': status}}
                    )
                    if update_result.modified_count > 0:
                        logger.info(f"Updated CreationChatStatus to '{status}'.")
                    else:
                        logger.warning(f"No documents were updated. The status might already be '{status}'.")
            else:
                logger.warning(f"No concept document found for enterprise_id: {enterprise_id_str} to update status.")
        except Exception as e:
            logger.error(f"Error updating CreationChatStatus: {str(e)}")

    async def update_ConceptCreationConvo(self, sender: str, content: str):
        logger.info(f"Updating concept creation convo with sender: {sender}, content: {content[:50]}...")
        try:
            # Force status 0 for first Concept_Refinement_Agent response
            if sender == 'Concept_Refinement_Agent':
                content = await self.ensure_initial_status_zero(content)
            
            message = {
                'timestamp': datetime.utcnow().isoformat(),
                'sender': sender,
                'content': content,
                'role': 'user' if sender == 'user_proxy' else 'assistant',
                'name': sender
            }

            # Check if this message is already in the history
            if not any(msg['content'] == content and msg['sender'] == sender for msg in self.AgentHistory):
                self.AgentHistory.append(message)

            # Ensure enterprise_id is string for concepts_collection
            enterprise_id_str = str(self.enterprise_id)
            
            # Fetch the latest concept document
            latest_concept = await concepts_collection.find_one(
                {"enterprise_id": enterprise_id_str},
                sort=[("ConceptCode", DESCENDING)]
            )
            
            if latest_concept:
                # Initialize update data with the message push
                update_data = {'$push': {'ConceptCreationConvo': message}}
                update_data['$set'] = {}

                # Extract data based on the agent type
                if sender == 'Users_Vision_Analyst':
                    try:
                        # Clean the message content
                        clean_message = content.replace('```json', '').replace('```', '').strip()
                        vision_data = json.loads(clean_message)

                        # Try to parse as VisionAgentOutputs
                        if isinstance(vision_data, dict):
                            # Handle both direct response and wrapped response
                            vision_response = (
                                vision_data.get('output_response', [{}])[0] 
                                if 'output_response' in vision_data 
                                else vision_data
                            )

                            # Map the fields to your database schema
                            update_data['$set'].update({
                                'CoreFocus': vision_response.get('core_focus', ''),
                                'ThirdPartyIntegrations': vision_response.get('third_party_integrations', []),
                                'SuggestedFeatures': vision_response.get('suggested_features', []),
                                'Monetization': vision_response.get('monetization', 0),
                                'LikedApps': vision_response.get('liked_apps', [])
                            })

                            logger.info(f"Extracted vision data: {update_data['$set']}")

                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse Users_Vision_Analyst output: {content}")
                    except Exception as e:
                        logger.error(f"Error processing Users_Vision_Analyst data: {str(e)}")

                elif sender == 'Concept_Refinement_Agent':
                    try:
                        # Clean the message content
                        clean_message = content.replace('```json', '').replace('```', '').strip()
                        concept_data = json.loads(clean_message)

                        # Try to parse as ConceptRefinementOutputs
                        if isinstance(concept_data, dict):
                            # Handle both direct response and wrapped response
                            concept_response = (
                                concept_data.get('output_response', [{}])[0] 
                                if 'output_response' in concept_data 
                                else concept_data
                            )

                            # Extract features in the correct format
                            features = concept_response.get('features', [])
                            formatted_features = []
                            for feature in features:
                                if isinstance(feature, dict):
                                    formatted_features.append({
                                        'feature_title': feature.get('feature_title', ''),
                                        'description': feature.get('description', '')
                                    })

                            # Map the fields to your database schema
                            update_data['$set'].update({
                                'Blueprint': {
                                    'Overview': concept_response.get('narrative', ''),
                                    'Features': formatted_features,
                                    'Tagline': concept_response.get('tagline', ''),
                                    'Note': concept_response.get('important_note', ''),
                                    'status': concept_response.get('status', 0)
                                }
                            })

                            logger.info(f"Extracted concept refinement data: {update_data['$set']}")

                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse Concept_Refinement_Agent output: {content}")
                    except Exception as e:
                        logger.error(f"Error processing Concept_Refinement_Agent data: {str(e)}")

                # Perform the database update
                if update_data['$set']:
                    result = await concepts_collection.update_one(
                        {'_id': latest_concept['_id']},
                        update_data
                    )
                    logger.info(f"Updated concept document. Modified count: {result.modified_count}")

            else:
                # Create new concept document if none exists
                new_ConceptCode = await self.get_next_ConceptCode()
                new_concept = {
                    "ConceptCode": new_ConceptCode,
                    "enterprise_id": enterprise_id_str,
                    "ConceptCreationConvo": [message],
                    "CoreFocus": "",
                    "ThirdPartyIntegrations": [],
                    "SuggestedFeatures": [],
                    "Monetization": 0,
                    "LikedApps": [],
                    "UsedTokens": {
                        "ConceptCreationAnalysis": {
                            "PromptTokens": {},
                            "CompletionTokens": {},
                            "TotalTokens": {}
                        }
                    },
                    "TotalCost": {
                        "ConceptCreationAnalysis": {}
                    },
                    "Blueprint": {
                        "Overview": "",
                        "Features": [],
                        "Tagline": "",
                        "Note": "",
                        "status": 0
                    }
                }
                
                result = await concepts_collection.insert_one(new_concept)
                logger.info(f"Created new concept document with id: {result.inserted_id}")

        except Exception as e:
            logger.error(f"Error updating concept creation convo: {str(e)}")
            logger.error(traceback.format_exc())

    async def get_next_ConceptCode(self):
        # Ensure enterprise_id is string for concepts_collection
        enterprise_id_str = str(self.enterprise_id)
        
        latest_concept = await concepts_collection.find_one(
            {"enterprise_id": enterprise_id_str, "ConceptCode": {"$exists": True}},
            sort=[("ConceptCode", DESCENDING)]
        )
        if latest_concept:
            return latest_concept["ConceptCode"] + 1
        else:
            return 1

    async def save_chat_state(self):
        try:
            # Ensure enterprise_id is string for concepts_collection
            enterprise_id_str = str(self.enterprise_id)
            
            latest_concept = await concepts_collection.find_one(
                {"enterprise_id": enterprise_id_str},
                sort=[("ConceptCode", DESCENDING)]
            )
            
            if latest_concept:
                # Retrieve the existing CreationChatState
                existing_chat_state = latest_concept.get('CreationChatState', {})

                # Prepare the new state to save
                state_to_save = {
                    'SessionID': self.SessionID,
                    'AgentHistory': self.AgentHistory,
                    'IterationCount': self.IterationCount,
                    'Concept_Refinement_Agent_count': self.Concept_Refinement_Agent_count,
                    'UserFeedbackCount': self.UserFeedbackCount,
                    'LastSpeaker': "Feedback_Agent"
                }
                
                # Preserve the existing SessionTotals
                if 'SessionTotals' in existing_chat_state:
                    state_to_save['SessionTotals'] = existing_chat_state['SessionTotals']



                # Update the document, merging the new state with existing data
                result = await concepts_collection.update_one(
                    {'_id': latest_concept['_id']},
                    {'$set': {
                        'CreationChatState': {
                            **existing_chat_state,  # Preserve existing data
                            **state_to_save  # Update with new data
                        }
                    }}
                )
                logger.info(f"Chat state updated in latest concept document. Modified count: {result.modified_count}")
            else:
                logger.warning("No concept document found to update chat state")

        except Exception as e:
            logger.error(f"Error saving chat state: {str(e)}")
            logger.error(traceback.format_exc())

    async def calculate_and_update_usage(self):
        try:
            # Gather usage summary for all agents
            agent_summary = gather_usage_summary(self.groupchat.agents)
            
            # Calculate session usage
            session_PromptTokens = 0
            session_CompletionTokens = 0
            session_TotalTokens = 0
            session_TotalCost = 0.0

            for Model_name, Model_data in agent_summary["usage_including_cached_inference"].items():
                if Model_name != 'total_cost':
                    session_PromptTokens += Model_data.get('prompt_tokens', 0)
                    session_CompletionTokens += Model_data.get('completion_tokens', 0)
                    session_TotalTokens += Model_data.get('total_tokens', 0)
                    session_TotalCost += Model_data.get('cost', 0.0)

            # Log the updates
            logger.info(f"Session usage - Prompt: {session_PromptTokens}, Completion: {session_CompletionTokens}, Total: {session_TotalTokens}, Cost: {session_TotalCost}")

            # Skip saving if all usage and costs are zero
            if session_TotalTokens == 0 and session_TotalCost == 0.0:
                logger.info("No usage or cost incurred. Skipping database update.")
                return

            # Update the database with session data
            try:
                await self.update_database_usage(
                    self.SessionID, 
                    session_PromptTokens, 
                    session_CompletionTokens, 
                    session_TotalTokens, 
                    session_TotalCost
                )
            except Exception as e:
                logger.error(f"Error updating database usage: {str(e)}")
                logger.error(traceback.format_exc())
                # You might want to add more specific error handling here if needed
        except Exception as e:
            logger.error(f"Error in calculate_and_update_usage: {str(e)}")
            logger.error(traceback.format_exc())

    async def update_database_usage(self, SessionID, session_PromptTokens, session_CompletionTokens, session_TotalTokens, session_TotalCost):
        logger.info(f"Updating database usage for enterprise_id: {self.enterprise_id}")
        try:
            # Convert self.enterprise_id to string if it's not already
            enterprise_id_str = str(self.enterprise_id)
            
            latest_concept = await concepts_collection.find_one(
                {"enterprise_id": enterprise_id_str},
                sort=[("ConceptCode", DESCENDING)]
            )
            
            if latest_concept:
                # Ensure 'SessionTotals' exists in 'CreationChatState'
                if 'CreationChatState' not in latest_concept or 'SessionTotals' not in latest_concept['CreationChatState']:
                    await concepts_collection.update_one(
                        {'_id': latest_concept['_id']},
                        {'$set': {
                            'CreationChatState.SessionTotals': {}
                        }}
                    )
                    # Refetch the document after update
                    latest_concept = await concepts_collection.find_one(
                        {"enterprise_id": enterprise_id_str},
                        sort=[("ConceptCode", DESCENDING)]
                    )

                # Calculate new tokens used in this interaction
                previous_session_data = latest_concept['CreationChatState']['SessionTotals'].get(f'session_{SessionID}', {})
                previous_TotalTokens = previous_session_data.get('TotalTokens', 0)
                new_tokens_used = session_TotalTokens - previous_TotalTokens

                # Update the current session data in CreationChatState
                session_data = {
                    'PromptTokens': session_PromptTokens,
                    'CompletionTokens': session_CompletionTokens,
                    'TotalTokens': session_TotalTokens,
                    'TotalCost': session_TotalCost
                }

                # Use $set to update the specific session without overwriting the entire SessionTotals
                await concepts_collection.update_one(
                    {'_id': latest_concept['_id']},
                    {'$set': {
                        f'CreationChatState.SessionTotals.session_{SessionID}': session_data,
                        'last_updated': datetime.utcnow()
                    }}
                )

                # Re-fetch the updated document to ensure we have the latest structure
                latest_concept = await concepts_collection.find_one(
                    {"enterprise_id": enterprise_id_str},
                    sort=[("ConceptCode", DESCENDING)]
                )
                
                # Calculate combined totals based on all session data
                combined_PromptTokens = sum(session.get('PromptTokens', 0) for session in latest_concept['CreationChatState']['SessionTotals'].values())
                combined_CompletionTokens = sum(session.get('CompletionTokens', 0) for session in latest_concept['CreationChatState']['SessionTotals'].values())
                combined_TotalTokens = sum(session.get('TotalTokens', 0) for session in latest_concept['CreationChatState']['SessionTotals'].values())
                combined_TotalCost = sum(session.get('TotalCost', 0.0) for session in latest_concept['CreationChatState']['SessionTotals'].values())

                # Update combined totals for creation analysis
                await concepts_collection.update_one(
                    {'_id': latest_concept['_id']},
                    {'$set': {
                        'UsedTokens.ConceptCreationAnalysis.PromptTokens': combined_PromptTokens,
                        'UsedTokens.ConceptCreationAnalysis.CompletionTokens': combined_CompletionTokens,
                        'UsedTokens.ConceptCreationAnalysis.TotalTokens': combined_TotalTokens,
                        'TotalCost.ConceptCreationAnalysis': combined_TotalCost
                    }}
                )

                # Calculate overall combined totals (verification + creation)
                verification_totals = latest_concept.get('UsedTokens', {}).get('ConceptVerificationAnalysis', {})
                overall_combined_totals = {
                    'PromptTokens': verification_totals.get('PromptTokens', 0) + combined_PromptTokens,
                    'CompletionTokens': verification_totals.get('CompletionTokens', 0) + combined_CompletionTokens,
                    'TotalTokens': verification_totals.get('TotalTokens', 0) + combined_TotalTokens
                }
                overall_combined_cost = (
                    latest_concept.get('TotalCost', {}).get('ConceptVerificationAnalysis', 0.0) +
                    combined_TotalCost
                )

                # Update overall combined totals
                await concepts_collection.update_one(
                    {'_id': latest_concept['_id']},
                    {'$set': {
                        'UsedTokens.CombinedAnalysis': overall_combined_totals,
                        'TotalCost.CombinedAnalysis': overall_combined_cost
                    }}
                )

                # Update the enterprise's token balance
                if new_tokens_used > 0:
                    try:
                        logger.info(f"Updating token balance for Enterprise_ID {enterprise_id_str}.")
                        result = await enterprises_collection.update_one(
                            {"_id": ObjectId(self.enterprise_id)},
                            {"$inc": {"AvailableTokens": -new_tokens_used}}
                        )
                
                        if result.modified_count == 0:
                            logger.warning(f"Failed to update token balance for enterprise {enterprise_id_str}")
                        else:
                            logger.info(f"Updated token balance for enterprise {enterprise_id_str}. Deducted {new_tokens_used} tokens.")
                    except Exception as e:
                        logger.error(f"Error updating enterprise token balance: {str(e)}")
                else:
                    logger.info(f"No new tokens used for enterprise {enterprise_id_str}. Skipping balance update.")

                logger.info("Updated combined totals with creation tokens and cost.")
            else:
                logger.warning(f"No concept document found for enterprise_id: {enterprise_id_str}")
        except Exception as e:
            logger.error(f"Error in update_database_usage: {str(e)}")
            logger.error(traceback.format_exc())

    async def display_token_usage(self):
        # Ensure enterprise_id is string for concepts_collection
        enterprise_id_str = str(self.enterprise_id)
        
        latest_concept = await concepts_collection.find_one(
            {"enterprise_id": enterprise_id_str},
            sort=[("ConceptCode", DESCENDING)]
        )
        if latest_concept:
            CreationChatState = latest_concept.get('CreationChatState', {})
            SessionTotals = CreationChatState.get('SessionTotals', {})

            if isinstance(SessionTotals, dict):
                # Display individual session totals
                for SessionID, data in SessionTotals.items():
                    print(f"Token Usage for Session (ID: {SessionID}):")
                    print(f"  Prompt Tokens: {data.get('PromptTokens', 0)}")
                    print(f"  Completion Tokens: {data.get('CompletionTokens', 0)}")
                    print(f"  Total Tokens: {data.get('TotalTokens', 0)}")
                    print(f"  Total Cost: ${data.get('TotalCost', 0.0):.5f}")

                # Display creation totals
                creation_totals = latest_concept.get('UsedTokens', {}).get('ConceptCreationAnalysis', {})
                if creation_totals:
                    print("\nCreation Totals Across All Sessions:")
                    print(f"  Creation Prompt Tokens: {creation_totals.get('PromptTokens', 0)}")
                    print(f"  Creation Completion Tokens: {creation_totals.get('CompletionTokens', 0)}")
                    print(f"  Creation Total Tokens: {creation_totals.get('TotalTokens', 0)}")
                    print(f"  Creation Total Cost: ${latest_concept.get('TotalCost', {}).get('ConceptCreationAnalysis', 0.0):.5f}")

                # Display overall combined totals
                combined_totals = latest_concept.get('UsedTokens', {}).get('CombinedAnalysis', {})
                if combined_totals:
                    print("\nOverall Combined Totals (Verification + Creation):")
                    print(f"  Combined Prompt Tokens: {combined_totals.get('PromptTokens', 0)}")
                    print(f"  Combined Completion Tokens: {combined_totals.get('CompletionTokens', 0)}")
                    print(f"  Combined Total Tokens: {combined_totals.get('TotalTokens', 0)}")
                    print(f"  Combined Total Cost: ${latest_concept.get('TotalCost', {}).get('CombinedAnalysis', 0.0):.5f}")
            else:
                print("No session totals data available or the structure is not a dictionary.")
        else:
            print(f"No token usage data available for enterprise_id: {enterprise_id_str}")

    async def send_to_websocket(self, sender: str, content: str):
        try:
            if isinstance(content, dict):
                content = content.get('content', '')

            # Ensure enterprise_id is string for concepts_collection
            enterprise_id_str = str(self.enterprise_id)
            
            latest_concept = await concepts_collection.find_one(
                {"enterprise_id": enterprise_id_str, "ConceptCode": {"$exists": True}},
                sort=[("ConceptCode", DESCENDING)]
            )

            CreationChatStatus = latest_concept.get('CreationChatStatus', {})

            BluePrintReport = latest_concept.get('Blueprint', {}) if latest_concept else {}

            await self.websocket.send_text(json.dumps({
                "sender": sender,
                "content": content,
                "CreationChatStatus": CreationChatStatus,
                "BluePrintReport": BluePrintReport
            }))
        except WebSocketDisconnect:
            logger.info(f"WebSocket already disconnected for {self.chat_id}. Skipping message send.")
        except Exception as e:
            logger.error(f"Error sending message to WebSocket: {str(e)}")

    async def resume_group_chat(self):
        try:
            logger.info("Attempting to resume group chat...")
            
            # Ensure enterprise_id is string for concepts_collection
            enterprise_id_str = str(self.enterprise_id)
            
            # Find the latest concept document for this enterprise
            latest_concept = await concepts_collection.find_one(
                {"enterprise_id": enterprise_id_str},
                sort=[("ConceptCode", DESCENDING)]
            )
          
            if latest_concept and 'CreationChatState' in latest_concept:
                saved_state = latest_concept['CreationChatState']
                
                self.AgentHistory = saved_state.get('AgentHistory', [])
                self.IterationCount = saved_state.get('IterationCount', 0)
                self.Concept_Refinement_Agent_count = saved_state.get('Concept_Refinement_Agent_count', 0)
                self.UserFeedbackCount = saved_state.get('UserFeedbackCount', 0)
                self.SessionID = saved_state.get('SessionID', str(uuid.uuid4()))


                # Set the last speaker based on the saved state
                LastSpeaker_name = saved_state.get('LastSpeaker')
                self.LastSpeaker = next((agent for agent in self.agents if agent.name == LastSpeaker_name), None)
                
                if not self.LastSpeaker:
                    logger.warning(f"Last speaker {LastSpeaker_name} not found in agents list. Defaulting to Feedback_Agent.")
                    self.LastSpeaker = next((agent for agent in self.agents if agent.name == "Feedback_Agent"), None)

                logger.info("Group chat resumed successfully")
                return True, self.LastSpeaker, self.AgentHistory[-1] if self.AgentHistory else None
            else:
                logger.info(f"No saved state found for enterprise_id: {enterprise_id_str}. Starting a new chat.")
                return False, None, None
        except Exception as e:
            logger.error(f"Error resuming group chat: {str(e)}")
            logger.error(traceback.format_exc())
            return False, None, None

    async def handle_disconnection(self):
        if not hasattr(self, '_disconnection_handled'):
            self._disconnection_handled = True
            try:
                logger.info(f"Handling disconnection for chat_id {self.chat_id}")

                # Ensure AgentHistory is accessed properly
                last_message = self.AgentHistory[-1] if self.AgentHistory else None

                current_agent = self.LastSpeaker
                while current_agent and hasattr(current_agent, 'name') and current_agent.name != "user_proxy":
                    next_agent = await self.state_transition(current_agent, self.groupchat_manager.groupchat, last_message)
                    if not next_agent:
                        logger.info("No next agent found. Ending chat flow.")
                        break

                    logger.info(f"Continuing chat with agent: {next_agent.name}")
                    try:
                        response = await asyncio.wait_for(
                            next_agent.a_respond(None, self.AgentHistory, current_agent, None),
                            timeout=30
                        )
                        if response and response[1]:
                            self.AgentHistory.append({
                                "role": "assistant", 
                                "content": response[1], 
                                "name": next_agent.name
                            })
                            logger.info(f"Response from {next_agent.name}: {response[1][:100]}...")  # Log first 100 chars

                            # Try to send to WebSocket; if it fails, skip sending
                            try:
                                await self.send_to_websocket(next_agent.name, response[1])
                            except WebSocketDisconnect:
                                logger.info(f"WebSocket disconnected for {self.chat_id}. Saving state and halting flow.")
                                await self.save_chat_state()
                                return  # End the flow here
                        else:
                            logger.info(f"No response from {next_agent.name}")
                    except asyncio.TimeoutError:
                        logger.warning(f"Timeout waiting for response from {next_agent.name}")
                    except Exception as e:
                        logger.error(f"Error getting response from {next_agent.name}: {str(e)}")

                    current_agent = next_agent
                    self.LastSpeaker = current_agent

                if current_agent and current_agent.name == "user_proxy":
                    logger.info(f"user_proxy reached, saving state and halting the chat flow.")
                    await self.save_chat_state()

                logger.info(f"WebSocket disconnected for {self.chat_id}. Chat state saved.")
            except Exception as e:
                logger.error(f"Error handling disconnection: {str(e)}")
                logger.error(traceback.format_exc())


    async def finalize_conversation(self):
        """Finalize the conversation and update the status in the database."""
        if not hasattr(self, '_conversation_finalized'):
            self._conversation_finalized = True
            logger.info("Finalizing the conversation.")

            try:
                # Ensure enterprise_id is string for concepts_collection
                enterprise_id_str = str(self.enterprise_id)
                
                # Find the latest concept document for this enterprise
                latest_concept = await concepts_collection.find_one(
                    {"enterprise_id": enterprise_id_str, "CreationChatStatus": 0},
                    sort=[("ConceptCode", DESCENDING)]
                )

                if latest_concept:
                    # Ensure the conversation status is updated to 1
                    current_status = latest_concept.get('CreationChatStatus', '')
                    if current_status == 0:
                        update_result = await concepts_collection.update_one(
                            {'_id': latest_concept['_id']},
                            {'$set': {'CreationChatStatus': 1}}
                        )
                        if update_result.modified_count > 0:
                            logger.info(f"Updated CreationChatStatus to 1 for concept code: {latest_concept['ConceptCode']}.")
                        else:
                            logger.warning("No documents were updated. The document might already be in 1 state.")
                    else:
                        logger.info(f"Chat already in '{current_status}' state. No update needed.")
                else:
                    logger.warning(f"No in-progress concept found for enterprise_id {enterprise_id_str} to finalize.")

            except Exception as e:
                logger.error(f"Error finalizing conversation: {str(e)}")
        
    def create_agents(self):
        user_proxy = UserProxyWebAgent(
            name="user_proxy",
            human_input_mode="ALWAYS",
            code_execution_config={"use_docker": False},
        )
        user_proxy.workflow_manager = self

        # Agent 1: Users_Vision_Analyst
        Users_Vision_Analyst = AsyncConversableWrapper(
            name="Users_Vision_Analyst",
            llm_config=llm_config_vision,
            system_message="""
            [ROLE] Vision Analyst
            [OBJECTIVE] Analyze the entire conversation with the user regarding their app idea and extract all critical information to form a clear, structured summary.
            
            [CONTEXT]
            A full conversation (referred to as the “User's Vision” conversation) where the user has described their app concept. This conversation may include ideas, inspirations, potential features, and references to existing applications or technologies.
            
            [GUIDELINES] You must follow these guidelines strictly for legal reasons. Do not stray from them:
            - Output Compliance: You must adhere to the specified "Output Structure" and its instructions. Do not include any additional commentary in your output. Do not forget the 'Important Note' at the end.
            - Avoid Using Names: Do not name the concept.
            - Read the complete conversation carefully.
            - Identify and capture only the essential details.
            - If an expected element is missing, use "None" for strings or an empty list for arrays.
            - Use clear, direct language without extraneous commentary.
            
            [INSTRUCTIONS]
            1. Extract the following elements from the User's Vision conversation:
                a. **CoreFocus**: What is the app’s primary purpose or category? (Examples: "budget tracking", "fitness coaching"). Provide a short answer (maximum 5 words).
                b. **Monetization**: Indicate whether the app is intended to generate revenue. Respond with "1" for revenue-generating intent, or "0" for personal use. Use integers.
                c. **SuggestedFeatures**: Extract all key features mentioned by the user.
                - **Each feature must be formatted as an object in an array.**
                - **Do not summarize multiple features in one object. Each feature must be its own object.**
                - **Follow the exact format:** Each feature must contain a `"FeatureTitle"` and `"Description"`.
                d. **ThirdPartyIntegrations**: Identify external services, APIs, or integrations.
                - **Each integration must be formatted as an object in an array.**
                - **Do not combine multiple technologies in a single object. Each integration must be its own object.**
                - **Follow the exact format:** Each integration must contain a `"TechnologyTitle"` and `"Description"`.
                e. **LikedApps**: Note any existing applications mentioned as inspiration.
            2. If any element is not mentioned in the conversation, output "None" for strings or an empty list for arrays as appropriate.
            3. Ensure that your output is well-structured, concise, and adheres to the JSON format.

            [OUTPUT FORMAT]
            {
                "CoreFocus": "[Extracted core focus (max 5 words)]",
                "Monetization": [0 or 1],
                "SuggestedFeatures": [
                    {
                        "FeatureTitle": "[Concise, appealing title]",
                        "Description": "[Compelling differentiation and user benefit, max 40 words]"
                    },
                    {
                        "FeatureTitle": "[Another feature title]",
                        "Description": "[Description of this feature, max 40 words]"
                    }
                    // Repeat for all features mentioned
                ],
                "ThirdPartyIntegrations": [
                    {
                        "TechnologyTitle": "[Name of Technology]",
                        "Description": "[How it will be used in the application, max 40 words]"
                    },
                    {
                        "TechnologyTitle": "[Another integration]",
                        "Description": "[How this integration will function, max 40 words]"
                    }
                    // Repeat for all integrations mentioned
                ],
                "LikedApps": ["App A", "App B"]
            }
            """,
            human_input_mode="NEVER",
        )
        Users_Vision_Analyst.workflow_manager = self

        # Agent 2: Trend_Insight_Agent
        Trend_Insight_Agent = AsyncConversableWrapper(
            name="Trend_Insight_Agent",
            llm_config=self.llm_config,
            system_message="""
            [ROLE] Market Analyst
            [OBJECTIVE]
            Identify, analyze, and articulate emerging market trends and gaps **explicitly** related to the user’s app vision. Your insights will directly shape the next stages of app concept development.

            [CONTEXT]
            You receive structured insights from **Users_Vision_Analyst**, which include:
            - **CoreFocus**: The app’s primary function or category.
            - **MonetizationIntent**: Whether the app is meant for revenue generation (1) or personal use (0).
            - **SuggestedFeatures**: A structured list of user-defined app features.
            - **ThirdPartyIntegrations**: Any APIs, services, or technologies explicitly mentioned for integration.
            - **LikedApps**: Existing applications cited by the user as inspirations.

            [GUIDELINES]
            - **All trends must be directly linked to CoreFocus, SuggestedFeatures, or ThirdPartyIntegrations.**
            - **MonetizationIntent must shape trend evaluation:**
                - If **MonetizationIntent = 1**, trends should include insights into revenue models, monetization strategies, and industry best practices.
                - If **MonetizationIntent = 0**, trends should emphasize usability, engagement, and adoption patterns rather than revenue generation.
            - If applicable, assess the **relevance and impact of ThirdPartyIntegrations** within trends.
            - When analyzing LikedApps, **assess their recent performance, feature releases, and market reception.**
            - Do not include generic industry trends; **each identified trend must be explicitly justified**.

            [INSTRUCTIONS]
            1. Analyze the user’s **CoreFocus** and **SuggestedFeatures** from Users_Vision_Analyst.
            2. Identify **3-5 emerging trends or market gaps** **directly related** to the app’s domain.
            3. Follow this structured methodology for identifying and validating trends:

            - **Step 1: Industry Reports & Recent Technological Advances**
                - Examine authoritative market reports, recent technological innovations, and regulatory changes.
                - Identify **shifts in user behavior, AI adoption, security enhancements, or new regulations** affecting this industry.
                - Clearly state how these changes impact the app's CoreFocus or SuggestedFeatures.

            - **Step 2: Competitor & LikedApp Analysis**
                - Analyze existing competitors, especially those listed under LikedApps.
                - Identify where competitors are thriving and where there are gaps.
                - Determine whether any **recent updates or industry shifts have affected LikedApps positively or negatively**.

            - **Step 3: Monetization & Adoption Considerations**
                - If **MonetizationIntent = 1**, evaluate **successful monetization models, pricing strategies, premium feature adoption, and subscription trends**.
                - If **MonetizationIntent = 0**, identify **non-monetary trends** such as personalization, open-source adoption, or viral growth strategies.

            - **Step 4: Evaluating ThirdPartyIntegrations**
                - Determine whether the provided ThirdPartyIntegrations align with any **rising industry needs or trends**.
                - If an integration is outdated or misaligned, **explicitly state its potential limitations**.

            4. For each trend, include:
            a. **Trend Title** (Max 6 words).
            b. **Detailed Explanation** (Max 100 words) covering:
                - **Why this trend is emerging** (**cite industry shifts, technology evolution, or user behavior changes**).
                - **The impact this trend will have on the market**.
                - **How this trend aligns with the user’s CoreFocus and SuggestedFeatures**.
                - **Explicit assessment of ThirdPartyIntegration relevance**, or state "No relevant integration identified."

            [OUTPUT FORMAT]
            - Trend 1: [Short title, max 6 words] – Explanation: [Detailed explanation, max 100 words]
            - Trend 2: [Short title] – Explanation: [Detailed explanation]
            ...
            """,
            human_input_mode="NEVER",
        )
        Trend_Insight_Agent.workflow_manager = self

        Divergent_Thinker_Agent = AsyncConversableWrapper(
            name="Divergent_Thinker_Agent",
            llm_config=self.llm_config,
            system_message="""
            [ROLE] Creative Ideator – Divergent Thinker

            [OBJECTIVE]
            Generate **bold, unconventional, and highly unique** app concepts based on insights from **Trend_Insight_Agent**, while leveraging aspects of the user’s original vision to inspire—but not constrain—creativity.

            [CONTEXT]
            You receive:
            - **Market trends and gaps from Trend_Insight_Agent** to ensure that concepts align with emerging opportunities.
            - **Structured insights from Users_Vision_Analyst** to maintain **some** alignment with the user’s initial vision, while still pushing creativity forward.  

            Specifically, you will reference:
            - **CoreFocus**: The app's general category. Your concepts should explore **unconventional interpretations** of this category.
            - **MonetizationIntent**: If the app is intended for revenue generation (**1**), consider how unique monetization models could emerge. If it is for personal use (**0**), focus on engagement and user-driven value.
            - **SuggestedFeatures**: These are **inspiration points, not restrictions**. You should challenge, invert, exaggerate, or combine them in unexpected ways.
            - **ThirdPartyIntegrations**: Consider whether these technologies inspire new, disruptive applications. If they are limiting innovation, **you may discard or rethink their use entirely**.

            [GUIDELINES]
            - **Every concept must directly connect to at least one market trend from Trend_Insight_Agent.**
            - **Concepts should challenge industry norms, break traditional thinking patterns, and present disruptive value.**
            - Avoid simply refining existing app models—**think beyond what currently exists**.
            - Use **advanced creativity methodologies** to break predictable ideation patterns.
            - Use proven ideation techniques such as **TRIZ, lateral thinking, biomimicry, and random stimulation** to generate **truly novel** ideas.
            - **Ignore feasibility concerns** at this stage—explore extreme ideas without constraints.

            [INSTRUCTIONS]
            1. **Analyze Trends for Inspiration**
            - Review the **3–5 key market trends** provided by **Trend_Insight_Agent**.
            - For each trend, ask:
                - How could this trend be taken to an **extreme**?
                - What **unexpected user behaviors** might emerge from this trend?
                - How could this trend be **disrupted or flipped** to create something radically different?

            2. **Reference (but do not be limited by) Users_Vision_Analyst's Insights**
            - Use **CoreFocus** as a **starting point**, but feel free to disrupt assumptions in this category.
            - Consider **SuggestedFeatures** as inspiration but aim to **invert, modify, or combine them in unexpected ways**.
            - For **MonetizationIntent**, challenge existing business models and explore **new monetization or engagement strategies**.
            - **Evaluate** whether ThirdPartyIntegrations fuel or limit creativity. If necessary, **completely rethink them**.

            3. **Apply Systematic Divergent Thinking Techniques**
            - Use the following structured ideation approaches:

            **SCAMPER Method (Substitute, Combine, Adapt, Modify, Put to Another Use, Eliminate, Reverse)**
            - Example: If a trend involves AI personalization, could we **reverse** it by giving users full control over AI decision-making?

            **Inversion Thinking ("What if we did the opposite?")**
            - Example: If a trend focuses on privacy, what if we made data sharing **completely transparent** in a beneficial way?

            **Cross-Industry Innovation (Apply solutions from different fields)**
            - Example: If gaming trends show high engagement, could those mechanics be applied to financial investing apps?

            **Extreme Constraints ("What if resources were extremely limited?")**
            - Example: If users had **no internet access**, how would the app still function effectively?

            4. **Generate 5–7 Radical App Concepts**
            - Each concept must introduce a fundamentally new approach, leveraging trends **in a non-obvious way**.
            - Each concept should **challenge industry norms** while remaining connected to user needs.

            5. **Format Each Concept as Follows**
            - a. **Title** (Max 5 words): A striking, memorable title capturing the essence of the idea.
            - b. **Description** (Max 40 words): A concise explanation highlighting:
                - **What makes this idea radically unique.**
                - **How it challenges conventional thinking.**
                - **Which market trend inspired it.**
                - **Any relevant inspiration from the user’s initial vision.**

            [OUTPUT FORMAT]
            1. Title: [Short, compelling title, max 5 words]
            Description: [Why this concept is uniquely innovative, max 40 words]
            2. Title: [Short, compelling title, max 5 words]
            Description: [Why this concept is uniquely innovative, max 40 words]
            
            (Continue until 5–7 concepts are generated)
            """,
            human_input_mode="NEVER",
        )
        Divergent_Thinker_Agent.workflow_manager = self

        # Agent 4: Concept_Evaluator_Agent
        Concept_Evaluator_Agent = AsyncConversableWrapper(
            name="Concept_Evaluator_Agent",
            llm_config=self.llm_config,
            system_message="""
            [ROLE] Concept Evaluator – Selection & Refinement

            [OBJECTIVE]
            Analyze and evaluate all app concepts generated by **Divergent_Thinker_Agent** and systematically **select the top 2 concepts** that demonstrate the best balance of **originality, feasibility, market potential, and alignment with the user's vision**.  
            Instead of providing separate refinement suggestions, you must **integrate improvements directly** into the concept descriptions so they are immediately ready for synthesis.

            [CONTEXT]
            You receive **5–7 unconventional app concepts** from **Divergent_Thinker_Agent** and must:
            - **Internally score** all concepts based on structured evaluation criteria.
            - **Narrow the selection to the top 2 concepts**, ensuring they are **highly unique, feasible, and market-ready**.
            - If necessary, **merge complementary concepts** to form stronger, more strategic ideas.
            - **Refine and enhance** the chosen concepts **before outputting them** to improve feasibility, market alignment, or clarity.

            Additionally, reference **structured insights from Users_Vision_Analyst**, including:
            - **CoreFocus**: The primary function/category of the app.
            - **MonetizationIntent**: Whether the user wants to generate revenue (**1**) or build a personal-use app (**0**).
            - **SuggestedFeatures**: Key features the user has expressed interest in.
            - **ThirdPartyIntegrations**: External services, APIs, or technologies the user wants to integrate.

            [EVALUATION CRITERIA]
            Score each concept across the following four categories:

            - **Originality (1-10):**  
            - How novel is this idea? Does it present a **fresh take on an existing problem**, or is it **completely unique**?  
            - Does it challenge conventional industry thinking or create a **new type of user experience**?  

            - **Feasibility (1-10):**  
            - Can this concept be realistically built using **current or near-future technology**?  
            - Are there major **technical, financial, or regulatory challenges** that would block execution?  
            - If feasibility is low, could **minor refinements** make it achievable?  

            - **Market Potential (1-10):**  
            - Is there a **clear demand** for this idea?  
            - Does it align with an **underserved or growing market segment**?  
            - Could this idea be **effectively monetized** if MonetizationIntent = 1?  

            - **Alignment with User’s Vision (1-10):**  
            - Does the concept **match or enhance the user’s CoreFocus**?  
            - Does it incorporate or **expand upon SuggestedFeatures** in a meaningful way?  
            - If ThirdPartyIntegrations are involved, are they **used strategically** or should they be reconsidered?  

            [INSTRUCTIONS]
            1. **Score each concept** internally using the evaluation criteria above.
            2. **Calculate a weighted final score** using the formula:  
            **Final Score = (Originality * 0.35) + (Feasibility * 0.25) + (Market Potential * 0.25) + (Alignment with Vision * 0.15)**  
            - **Originality is weighted the highest (35%)** to prioritize breakthrough ideas.  
            - **Feasibility and Market Potential are weighted at 25%** each to ensure realistic execution.  
            - **Alignment with Vision is weighted at 15%** to ensure the concept is still tailored to user preferences.  
            3. **Select the top 2 highest-scoring concepts** to proceed.  
            - If two concepts are **closely related**, **merge them into a stronger, more viable concept**.  
            4. **Refine and enhance the selected concepts** by:  
            - Improving clarity, feasibility, or market positioning.  
            - Adjusting feature sets to better align with **SuggestedFeatures** and **ThirdPartyIntegrations**.  
            - Making strategic **optimizations** based on evaluation feedback.  
            5. **Format the output with fully refined concept descriptions** that include these enhancements.

            [OUTPUT FORMAT]
            Provide exactly **2 selected concepts**, fully refined and ready for synthesis:

            {
                "SelectedConcepts": [
                    {
                        "Title": "[Final Concept Title]",
                        "FinalScore": [Weighted score],
                        "Description": "[Fully refined and enhanced concept description, max 100 words]"
                    },
                    {
                        "Title": "[Final Concept Title]",
                        "FinalScore": [Weighted score],
                        "Description": "[Fully refined and enhanced concept description, max 100 words]"
                    }
                ]
            }
            """,
            human_input_mode="NEVER",
        )
        Concept_Evaluator_Agent.workflow_manager = self

        # Agent 5: Idea_Synthesizer_Agent
        Idea_Synthesizer_Agent = AsyncConversableWrapper(
            name="Idea_Synthesizer_Agent",
            llm_config=self.llm_config,
            system_message="""
            [ROLE] Innovation Architect

            [OBJECTIVE]
            Transform the **two strongest app concepts** from **Concept_Evaluator_Agent** into a **single, unified, market-ready app concept**. Your goal is to **merge the best elements**, eliminate redundancies, and ensure the final concept is both **highly innovative and feasible**.

            [CONTEXT]
            You receive **exactly two refined app concepts** from **Concept_Evaluator_Agent**, each containing:
            - **Title** (Concise concept name)
            - **FinalScore** (Weighted evaluation score based on originality, feasibility, market potential, and alignment with user vision)
            - **Description** (Fully refined concept, with feasibility enhancements already integrated)

            You also receive structured insights from **Users_Vision_Analyst**, including:
            - **CoreFocus** (The app’s primary function or category)
            - **MonetizationIntent** (Whether the app is meant for revenue generation or personal use)
            - **SuggestedFeatures** (User-defined features that were initially envisioned)
            - **ThirdPartyIntegrations** (External services, APIs, or technologies referenced by the user)

            [GUIDELINES]
            - **Do not simply list or stack features**—intelligently merge and refine them into a **stronger, more coherent app concept**.
            - **Ensure alignment with the user's CoreFocus** while enhancing feasibility.
            - **Remove redundant or conflicting features** to streamline the final product.
            - **If ThirdPartyIntegrations are mentioned**, use them **only where they add clear strategic value**.
            - **Ensure a clear user purpose, business potential, and technological viability**.

            [INSTRUCTIONS]
            1. **Analyze the two refined concepts** from Concept_Evaluator_Agent.
            - Identify **overlapping strengths** that can be **merged into a more effective solution**.
            - If a feature from one concept enhances the other, **integrate it strategically**.
            - Remove **any conflicting, redundant, or impractical elements**.

            2. **Use insights from Users_Vision_Analyst** to align the concept with the user’s vision:
            - Ensure the final concept **aligns with CoreFocus** and **enhances the user’s original intent**.
            - If MonetizationIntent = 1, ensure **the concept has a viable revenue model**.
            - Integrate **SuggestedFeatures** where appropriate to ensure the concept remains aligned with user expectations.
            - Use **ThirdPartyIntegrations** **only if they enhance the app’s functionality**—do not force them into the concept if they do not fit.

            3. **Synthesize a single, cohesive app concept** that:
            - **Captures the strongest elements** from both ideas.
            - **Maintains a clear problem-solution framework** to ensure user value.
            - **Enhances feasibility** by simplifying complex features where necessary.
            - **Aligns with market trends and user needs** while retaining originality.

            4. **Create a market-ready concept description**:
            - Clearly articulate the **problem it solves** and the **unique value proposition**.
            - Emphasize the **innovative aspects** while ensuring **practical execution**.
            - If applicable, highlight **how third-party integrations enhance functionality**.
            - Keep the description **concise, persuasive, and clear**.

            [OUTPUT FORMAT]
            {
                "SynthesizedConceptTitle": "[Unified concept title, max 6 words]",
                "Description": "[Clear, innovative, and cohesive concept overview, max 100 words]"
            }
            """,
            human_input_mode="NEVER",
        )
        Idea_Synthesizer_Agent.workflow_manager = self

        # Agent 6: Feature_Ideation_Agent
        Feature_Ideation_Agent = AsyncConversableWrapper(
            name="Feature_Ideation_Agent",
            llm_config=self.llm_config,
            system_message="""
            [ROLE] Feature Specialist

            [OBJECTIVE]
            Identify and define **distinct, high-impact features** for the synthesized app concept provided by **Idea_Synthesizer_Agent**. Each feature must clearly enhance the app's differentiation, market appeal, and user adoption potential.

            [CONTEXT]
            You receive the final, unified app concept from **Idea_Synthesizer_Agent**, along with structured insights from **Users_Vision_Analyst**, including:
            - **CoreFocus** (the app’s primary purpose/category)
            - **MonetizationIntent** (1 for revenue-generating intent, 0 for personal use)
            - **SuggestedFeatures** (features explicitly requested by the user)
            - **ThirdPartyIntegrations** (external APIs or technologies requested by the user)

            [GUIDELINES]
            - **Every feature must directly support** the synthesized concept and clearly relate to the user’s original vision (**CoreFocus & SuggestedFeatures**).
            - **Strategically incorporate ThirdPartyIntegrations** where they meaningfully enhance the user experience—avoid forcing unnecessary integrations.
            - Ensure **each feature distinctly sets the app apart** from existing competitors and current market standards.
            - Features should be innovative yet **technically realistic** to implement in the next 3-5 years.

            [INSTRUCTIONS]
            Follow these steps carefully when identifying features:

            **Step 1: Analyze Synthesized Concept**
            - Understand the **core value proposition and innovation** in the synthesized concept.

            **Step 2: Reference User Vision Insights**
            - Review **CoreFocus** and **SuggestedFeatures** from the Users_Vision_Analyst.
            - Identify opportunities to incorporate **ThirdPartyIntegrations** to amplify feature impact.

            **Step 3: Generate & Differentiate Features**
            - Define **3–5 highly differentiated core features** that strongly reinforce the app’s value and uniqueness.
            - Avoid generic or obvious features that competitors already widely implement.
            - Clearly articulate how each feature uniquely benefits the user, creating a competitive advantage.

            **For each feature, provide:**
            - **FeatureTitle**: Concise and appealing, clearly highlighting the benefit (max 5 words).
            - **Description**: Brief, compelling explanation of how the feature specifically benefits the user and differentiates the app from competitors. Explicitly mention if the feature strategically leverages any user-requested ThirdPartyIntegrations. (max 40 words).

            [OUTPUT FORMAT]
            {
                "CoreFeatures": [
                    {
                        "FeatureTitle": "[Concise, appealing title, max 5 words]",
                        "Description": "[Differentiating benefit clearly articulated, including strategic integration if applicable, max 40 words]"
                    },
                    {
                        "FeatureTitle": "[Concise, appealing title]",
                        "Description": "[Compelling differentiation and user benefit, max 40 words]"
                    },
                    ... (3–5 features total)
                ]
            }
            """,
            human_input_mode="NEVER",
        )
        Feature_Ideation_Agent.workflow_manager = self

        Concept_Refinement_Agent = AsyncConversableWrapper(
            name="Concept_Refinement_Agent",
            llm_config=llm_config_concept,
            system_message="""
            [ROLE] Vision Architect & Concept Marketer

            [OBJECTIVE]
            Transform the unified app concept and associated features into a **persuasive, emotionally engaging presentation**, clearly highlighting the groundbreaking value and real-world benefits to inspire and excite the user.

            [CONTEXT]
            You receive the following inputs:
            - **SynthesizedConceptTitle**: Unified app concept title (from Idea_Synthesizer_Agent).
            - **Description**: Detailed overview of the unified app concept (from Idea_Synthesizer_Agent).
            - **CoreFeatures**: A structured list of distinct, compelling features (from Feature_Ideation_Agent), each including:
            - **FeatureTitle** (short, descriptive title)
            - **Description** (clear, user-focused benefit with any technology explicitly named)

            [GUIDELINES] 
            You must follow these guidelines strictly for legal reasons. Do not stray from them:
            Output Compliance: You must adhere to the specified "Output Structure" and its instructions. Do not include any additional commentary in your output. Do not forget the 'Important Note' at the end.
            Avoid Using Names: Do not name the concept.
            Intial Status: You are part of an iterative refinement process, for your FIRST output in the conversation, ALWAYS set "status": 0 regardless of what information you've received from user_proxy.

            User Feedback Handling: 
            - ONLY set "status": 1 when responding to actual human user feedback AFTER your first response.
            - If the user provides porsitive feedback and doesn't request changes, THEN set "status": 1.
            - If the user requests further adjustments, keep "status": 0.
            - NEVER set "status": 1 in your first response, even if you received all the required context.'

            Instructions
            **Step 1: Crafting the Tagline**
            - **Directly reference and distill** the essence from the provided **SynthesizedConceptTitle and Description**.
            - Create a concise, emotionally charged, and memorable tagline (10-15 words).
            - Clearly highlight the concept’s innovative, transformative nature and the primary value it delivers.
            - Use dynamic, action-oriented language designed to immediately captivate interest.

            **Step 2: Crafting the Visionary Narrative**
            - Reread the **SynthesizedConceptTitle and Description** from the Idea_Synthesizer_Agent.
            - Write a visionary, captivating narrative (500-600 characters maximum), explicitly based on this synthesized concept:
                - Clearly articulate the primary user problem the concept solves.
                - Explicitly state the transformative user benefits.
                - Highlight the innovative approach or technology (if explicitly stated in the Description).
                - Create emotional resonance by vividly illustrating how the app positively impacts the user's life or work.
                - Avoid detailed lists of features—your narrative should provide a holistic, emotionally engaging view that sparks excitement and curiosity.

            **Step 3: Feature Highlights**
            - For each core feature provided by **Feature_Ideation_Agent**:
                - Create a compelling and concise **Feature Title** (max 5 words), emotionally evocative and action-driven.
                - Develop a clear, emotionally appealing **Feature Description** (max 40 words), explicitly referencing the provided technology (if any), clearly communicating the unique benefit to users, and highlighting differentiation from existing solutions.

            **Step 4: Handling User Feedback & Refinement**
            - You are part of an iterative refinement process, on your **initial generation**, always set `"status": 0`.
            - On **subsequent interactions**:
                - Carefully review any user feedback provided:
                    - Identify clearly whether the feedback calls for:
                        - **New features**,
                        - **Changes or removals to existing features**, or
                        - **Adjustments in the narrative or thematic focus**.
                    - If feedback is vague ("make it better") or nonsensical, make only minor logical adjustments while maintaining coherence.
                    - **Do NOT** alter the tagline or narrative significantly unless the feedback explicitly demands a core shift.
                - Set `"status": 1` **only if feedback explicitly confirms approval** or clearly indicates no further adjustments are required; otherwise, continue with `"status": 0`.

            [OUTPUT FORMAT]
            {
                "tagline": "Powerful, emotional tagline explicitly derived from synthesized concept (10-15 words)",
                "narrative": "Visionary, emotionally evocative narrative explicitly based on synthesized concept, clearly highlighting core value and transformative benefits (500-600 characters)",
                "features": [
                    {
                        "feature_title": "Concise, emotional title (max 5 words)",
                        "description": "Persuasive, differentiated description explicitly mentioning provided technologies and clearly stating user benefit while providing just enough tehcnical description for downstream agents without overwhelming the user (max 40 words)"
                    }
                    // Repeat for each provided feature
                ],
                "important_note": "While not all concepts require legal review, understanding potential regulations can help you avoid roadblocks. We recommend consulting a legal professional if you have any questions.",
                "status": 0  // Set to 1 only after explicit positive feedback, otherwise keep as 0.
            }
            """,
            human_input_mode="NEVER",
        )
        Concept_Refinement_Agent.workflow_manager = self

        Feedback_Agent = AsyncConversableWrapper(
            name="Feedback_Agent",
            llm_config=self.llm_config,
            system_message="""
            Role: Concept Feedback Iterative Agent
            Goal: Request user feedback iteratively regarding the concept you have created.
            
            Guidelines: You must follow these guidelines strictly for legal reasons. Do not stray from them:
            Do not deviate from the 2 provided questions.
            Use the exact questions without rephrasing or altering them.
            Ask one question at a time in the given order.
            Do not change the sequence of these questions.
            
            Instructions:
            1. Exact Questions and Order to Ask: 
            Does this concept spark your creativity? We want to know if it aligns with your overall vision for the project.
            The concept has been refined based on your feedback. Does it truly capture the essence of your vision?

            Output Structure:
            (The question to the user)
            """,
            max_consecutive_auto_reply=2,
        )
        Feedback_Agent.workflow_manager = self

        agents = [
            user_proxy, Users_Vision_Analyst, Trend_Insight_Agent,
            Divergent_Thinker_Agent, Concept_Evaluator_Agent, Idea_Synthesizer_Agent,
            Feature_Ideation_Agent, Concept_Refinement_Agent, Feedback_Agent
        ]

        return agents

    def create_groupchat(self):
        return autogen.GroupChat(
            agents=self.agents,
            messages=[],
            max_round=50,
            speaker_selection_method=self.state_transition,
        )

    def create_groupchat_manager(self):
        return AsyncGroupChatManager(
            groupchat=self.groupchat, 
            llm_config=self.llm_config,  # Pass unified config
            workflow_manager=self, 
            websocket=self.websocket
        )

    async def state_transition(self, LastSpeaker, groupchat, last_message):
        # Determine the name of the last speaker
        if isinstance(LastSpeaker, str):
            last_name = LastSpeaker
        elif LastSpeaker is None:
            last_name = "user_proxy"
        else:
            last_name = LastSpeaker.name

        logger.info(f"State transition: Last speaker was {last_name}")
        
        # Define enterprise_id_str at the beginning of the function to fix scope issues
        enterprise_id_str = str(self.enterprise_id)

        # Check for status in Concept_Refinement_Agent's response with improved error handling
        if last_name == "Concept_Refinement_Agent":
            try:
                # Try to parse the message as JSON if it's a string
                status = None
                
                # First check direct message content
                if isinstance(last_message, str):
                    # Only attempt to parse as JSON if it looks like JSON
                    if last_message.strip().startswith('{') and last_message.strip().endswith('}'):
                        try:
                            message_content = json.loads(last_message)
                            if isinstance(message_content, dict):
                                status = message_content.get('status')
                        except json.JSONDecodeError:
                            logger.debug(f"Message is not valid JSON despite appearance: {last_message[:100]}...")
                    # If not JSON-formatted, skip parsing attempt
                elif isinstance(last_message, dict):
                    status = last_message.get('status')
                    if status is None and 'output_response' in last_message:
                        # Handle nested structure
                        if last_message['output_response'] and isinstance(last_message['output_response'][0], dict):
                            status = last_message['output_response'][0].get('status')
                
                logger.info(f"Concept_Refinement_Agent direct status: {status}")
                
                # If no status directly, check inside BluePrintReport
                if status is None:
                    # Fetch the latest concept
                    latest_concept = await concepts_collection.find_one(
                        {"enterprise_id": enterprise_id_str},
                        sort=[("ConceptCode", DESCENDING)]
                    )
                    if latest_concept and 'Blueprint' in latest_concept:
                        blueprint_status = latest_concept['Blueprint'].get('status')
                        logger.info(f"Concept_Refinement_Agent BluePrintReport status: {blueprint_status}")
                        status = blueprint_status
                
                if status == 1:  # Status 1 indicates completion
                    logger.info("Status 1 received from Concept_Refinement_Agent. Finalizing the session.")
                    await self.finalize_conversation()
                    return None

            except Exception as e:
                logger.error(f"Error checking Concept_Refinement_Agent status: {str(e)}")
                logger.error(traceback.format_exc())

        # Rest of your existing state transition logic...
        if not hasattr(self, "main_loop_completed"):
            sequence = {
                "user_proxy": "Users_Vision_Analyst",
                "Users_Vision_Analyst": "Trend_Insight_Agent",
                "Trend_Insight_Agent": "Divergent_Thinker_Agent",
                "Divergent_Thinker_Agent": "Concept_Evaluator_Agent",
                "Concept_Evaluator_Agent": "Idea_Synthesizer_Agent",
                "Idea_Synthesizer_Agent": "Feature_Ideation_Agent",
                "Feature_Ideation_Agent": "Concept_Refinement_Agent",
                "Concept_Refinement_Agent": "Feedback_Agent",
                "Feedback_Agent": "user_proxy"
            }
            if last_name == "Concept_Refinement_Agent":
                self.main_loop_completed = True
        else:
            sequence = {
                "Concept_Refinement_Agent": "Feedback_Agent",
                "Feedback_Agent": "user_proxy",
                "user_proxy": "Concept_Refinement_Agent"
            }

        next_name = sequence.get(last_name, "user_proxy")
        next_agent = self.agent_dict.get(next_name)
        
        # If the next agent would be Feedback_Agent but we already have a status 1,
        # return None to signal the end of the conversation
        if next_agent and next_agent.name == "Feedback_Agent":
            # Check BluePrintReport status again
            latest_concept = await concepts_collection.find_one(
                {"enterprise_id": enterprise_id_str},
                sort=[("ConceptCode", DESCENDING)]
            )
            if latest_concept and latest_concept.get('Blueprint', {}).get('status') == 1:
                logger.info("Status 1 detected in BluePrintReport when transitioning to Feedback_Agent. Ending conversation.")
                await self.finalize_conversation()
                return None
        
        logger.info(f"State transition: Next speaker will be {next_agent.name if next_agent else 'None'}")
        self.LastSpeaker = next_agent
        return next_agent

    async def ensure_initial_status_zero(self, message):
        """Ensure the initial response from Concept_Refinement_Agent has status 0."""
        try:
            if isinstance(message, str):
                try:
                    message_data = json.loads(message)
                    # Check if this is the first Concept_Refinement_Agent message
                    if not hasattr(self, 'concept_refinement_count'):
                        self.concept_refinement_count = 0
                    
                    self.concept_refinement_count += 1
                    
                    if self.concept_refinement_count == 1:
                        logger.info("First response from Concept_Refinement_Agent - ensuring status is 0")
                        # Set status to 0 for the first response
                        if 'status' in message_data:
                            message_data['status'] = 0
                        elif 'output_response' in message_data and message_data['output_response']:
                            for response in message_data['output_response']:
                                if isinstance(response, dict) and 'status' in response:
                                    response['status'] = 0

                        # Ensure enterprise_id is string for concepts_collection
                        enterprise_id_str = str(self.enterprise_id)
                        
                        # Update the Blueprint in the database
                        await concepts_collection.update_one(
                            {"enterprise_id": enterprise_id_str},
                            {"$set": {"Blueprint.status": 0}},
                            sort=[("ConceptCode", DESCENDING)]
                        )
                        
                        return json.dumps(message_data)
                except json.JSONDecodeError:
                    pass
            
            return message
        except Exception as e:
            logger.error(f"Error ensuring initial status zero: {str(e)}")
            return message

    async def get_creation_chat_status(self):
        """Get the creation chat status from the database."""
        try:
            # Ensure enterprise_id is string for concepts_collection
            enterprise_id_str = str(self.enterprise_id)
            
            # Find the latest concept document for this enterprise
            latest_concept = await concepts_collection.find_one(
                {"enterprise_id": enterprise_id_str},
                sort=[("ConceptCode", DESCENDING)]
            )
            return latest_concept.get('CreationChatStatus', 0) if latest_concept else 0
        except Exception as e:
            logger.error(f"Error getting creation chat status: {str(e)}")
            return 0

class AsyncGroupChatManager:
    def __init__(self, websocket: WebSocket, groupchat, llm_config, workflow_manager: WorkflowManager):
        self.groupchat = groupchat
        self.websocket = websocket
        self.workflow_manager = workflow_manager
        self.autogengroupchatmanager = autogen.GroupChatManager(
            groupchat=self.groupchat, 
            llm_config=llm_config  # Use unified config
        )
        self.client_cache = {}

    async def a_send(self, message: str, sender: Union[AsyncConversableWrapper, UserProxyWebAgent]):
        try:
            original_sender = sender.name

            if isinstance(message, dict) and "name" in message:
                original_sender = message["name"]
                message = message.get("content", "")

            logger.info(f"AsyncGroupChatManager: Sending message from {original_sender}")
            logger.info(f"Message content: {message[:100]}...")  # Log first 100 characters of the message

            # Ensure message has a valid 'name'
            if not original_sender:
                original_sender = "unknown"  # Default to 'unknown' if the name is empty or missing

            await self.workflow_manager.update_ConceptCreationConvo(original_sender, message)
            
            # Save the chat state after each message
            await self.workflow_manager.save_chat_state()

            # Calculate and update token usage and cost after each interaction
            await self.workflow_manager.calculate_and_update_usage()

            workflow_manager_enterprise_id_str = str(self.workflow_manager.enterprise_id)
            # Fetch the latest concept document to get the Blueprint
            latest_concept = await concepts_collection.find_one(
                {"enterprise_id": workflow_manager_enterprise_id_str, "ConceptCode": {"$exists": True}},  
                sort=[("ConceptCode", DESCENDING)]
            )

            BluePrintReport = latest_concept.get('Blueprint', {}) if latest_concept else {}
            blueprint_status = BluePrintReport.get('status', 0)
            CreationChatStatus = latest_concept.get('CreationChatStatus', 0) if latest_concept else 0

            # Handle messages based on the sender
            if original_sender == "Feedback_Agent":
                # Check if we should even proceed to Feedback_Agent based on BluePrintReport status
                if blueprint_status == 1:
                    logger.info("Status 1 detected in BluePrintReport. Skipping Feedback_Agent and finalizing conversation.")
                    await self.workflow_manager.finalize_conversation()
                    
                    if self.websocket:
                        await self.websocket.send_text(json.dumps({
                            "sender": "system",
                            "content": "",
                            "CreationChatStatus": CreationChatStatus,
                            "BluePrintReport": BluePrintReport
                        }))
                    return True
                
                if self.websocket:
                    await self.workflow_manager.send_to_websocket(original_sender, message)
                else:
                    logger.error("WebSocket is not available to send message.")

            elif original_sender == "Concept_Refinement_Agent":
                try:
                    # First check direct message content
                    status = None
                    if isinstance(message, str):
                        try:
                            message_content = json.loads(message)
                            if isinstance(message_content, dict):
                                status = message_content.get('status')
                        except json.JSONDecodeError:
                            pass
                    elif isinstance(message, dict):
                        status = message.get('status')
                    
                    # If no status directly in message, check BluePrintReport
                    if status is None:
                        status = blueprint_status
                    
                    logger.info(f"Concept_Refinement_Agent status: direct={status}, blueprint={blueprint_status}")

                    if status == 1:
                        # Finalize the conversation if status is 1
                        await self.workflow_manager.finalize_conversation()
                        
                        # Get updated CreationChatStatus without using the method
                        updated_concept = await concepts_collection.find_one(
                            {"enterprise_id": workflow_manager_enterprise_id_str},
                            sort=[("ConceptCode", DESCENDING)]
                        )
                        updated_status = updated_concept.get('CreationChatStatus', 0) if updated_concept else 0

                        if self.websocket:
                            await self.websocket.send_text(json.dumps({
                                "sender": "system",
                                "content": "",
                                "CreationChatStatus": updated_status,
                                "BluePrintReport": BluePrintReport
                            }))
                        logger.info("Status 1 received. Ending chat.")
                        return True            
                    else:
                        if self.websocket:
                            await self.websocket.send_text(json.dumps({
                                "sender": "system",
                                "content": "",
                                "CreationChatStatus": CreationChatStatus,
                                "BluePrintReport": BluePrintReport
                            }))
                except Exception as e:
                    logger.error(f"Error processing Concept_Refinement_Agent message: {str(e)}")
                    logger.error(traceback.format_exc())

            # State transition logic
            next_agent = await self.workflow_manager.state_transition(sender, self.groupchat, message)
            logger.info(f"State transition returned next_agent: {next_agent.name if next_agent else 'None'}")

            # If next_agent is None, this means the state transition has detected a termination condition
            if next_agent is None:
                logger.info("State transition returned None, indicating chat termination.")
                return True

            if next_agent:
                logger.info(f"AsyncGroupChatManager: Next speaker is {next_agent.name}")
                response = await next_agent.a_respond(None, self.workflow_manager.AgentHistory, sender, None)
                if response and response[1]:
                    logger.info(f"Received response from {next_agent.name}: {response[1][:100]}...")
                    await self.a_send(response[1], next_agent)
                elif isinstance(next_agent, UserProxyWebAgent):
                    logger.info("UserProxyWebAgent did not return a response. Continuing chat flow.")
                else:
                    logger.info(f"{next_agent.name} did not return a valid response. Continuing chat flow.")

            return False  # Continue the chat flow

        except WebSocketDisconnect:
            logger.info("WebSocket disconnected. Saving chat state before halting.")
            await self.workflow_manager.save_chat_state()
            return True
        except Exception as e:
            logger.error(f"Error in a_send: {str(e)}")
            logger.error(traceback.format_exc())
            await self.workflow_manager.handle_disconnection()
            return True

    async def a_receive(self, message: str, sender: Union[AsyncConversableWrapper, UserProxyWebAgent]):
        logger.info(f"AsyncGroupChatManager: Received message from {sender.name}")
        return await self.a_send(message, sender)

class AutoGenChat:
    def __init__(self, chat_id: str, websocket: WebSocket, llm_config, client, user_id: str, enterprise_id: str):
        self.chat_id = chat_id
        self.websocket = websocket
        self.client = client
        self.enterprise_id = enterprise_id  # Store the enterprise_id for future use
        self.workflow_manager = WorkflowManager(
            llm_config=llm_config,
            websocket=websocket,
            chat_id=chat_id,
            user_id=user_id,
            enterprise_id=enterprise_id,
            client=client
        )
        self.max_resume_attempts = 5

    async def run_chat(self):
        try:
            # Ensure enterprise_id is string for concepts_collection
            workflow_manager_enterprise_id_str = str(self.workflow_manager.enterprise_id)
            # Fetch the latest concept document to get the Blueprint
            latest_concept = await concepts_collection.find_one(
                {"enterprise_id": workflow_manager_enterprise_id_str},
                sort=[("ConceptCode", DESCENDING)]
            )

            if not latest_concept:
                logger.error(f"No concept found for enterprise_id: {workflow_manager_enterprise_id_str}. This is unexpected as a concept should exist from the verification flow.")
                return

            # Check if BluePrintReport already has status 1
            BluePrintReport = latest_concept.get('Blueprint', {}) 
            if BluePrintReport and BluePrintReport.get('status') == 1:
                logger.info("BluePrintReport already has status 1. Finalizing conversation.")
                await self.workflow_manager.finalize_conversation()
                CreationChatStatus = latest_concept.get('CreationChatStatus', 0)
                if CreationChatStatus != 1:
                    logger.info("CreationChatStatus is not 1. Updating to 1.")
                    await concepts_collection.update_one(
                        {'_id': latest_concept['_id']},
                        {'$set': {'CreationChatStatus': 1}}
                    )
                
                if self.websocket:
                    await self.websocket.send_text(json.dumps({
                        "sender": "system",
                        "content": "Concept creation already completed successfully.",
                        "CreationChatStatus": 1,
                        "BluePrintReport": BluePrintReport
                    }))
                return

            resumed = False
            chat_ended = False
            last_agent = None
            last_message = None
            last_message_sent = False

            logger.info(f"Found concept for enterprise_id: {workflow_manager_enterprise_id_str}, ConceptCode: {latest_concept['ConceptCode']}")
            
            if latest_concept.get('CreationChatStatus') == 0:
                logger.info(f"Attempting to resume group chat for chat_id: {self.chat_id}")

                saved_state = latest_concept.get('CreationChatState', {})

                if saved_state:
                    logger.info(f"Resuming creation chat for chat_id {self.workflow_manager.chat_id}...")

                    self.workflow_manager.AgentHistory = saved_state.get('AgentHistory', [])
                    self.workflow_manager.IterationCount = saved_state.get('IterationCount', 0)
                    self.workflow_manager.Concept_Refinement_Agent_count = saved_state.get('Concept_Refinement_Agent_count', 0)
                    self.workflow_manager.UserFeedbackCount = saved_state.get('UserFeedbackCount', 0)
                    self.workflow_manager.SessionID = saved_state.get('SessionID', str(uuid.uuid4()))
                    self.workflow_manager.LastSpeaker = saved_state.get('LastSpeaker', "Feedback_Agent")

                    converted_data = [
                        {
                            "content": item.get("content", ""),
                            "role": item.get("role", ""),
                            "name": item.get("sender", "")
                        }
                        for item in self.workflow_manager.AgentHistory
                    ]

                    try:
                        last_agent, last_message = self.workflow_manager.autogengroupchatmanager.resume(messages=converted_data)
                    except Exception as e:
                        logger.error(f"Error resuming group chat: {str(e)}")

                    if last_agent and last_agent.name in ["Feedback_Agent", "user_proxy"]:
                        self.workflow_manager.initialize_new_session_for_tracking()
                        self.workflow_manager.LastSpeaker = last_agent
                        logger.info(f"Resumed creation chat with last speaker: {last_agent.name}")
                        resumed = True

                        await self.send_chat_history(include_last=False)

                        if last_message:
                            last_message_content = last_message.get('content', '') if isinstance(last_message, dict) else last_message
                            
                            try:
                                # Only try to parse as JSON if the last speaker wasn't Feedback_Agent
                                if last_agent.name != "Feedback_Agent":
                                    message_data = json.loads(last_message_content) if isinstance(last_message_content, str) else last_message_content
                                    status = message_data.get('status', 0)
                                else:
                                    # For Feedback_Agent messages, don't try to parse JSON
                                    status = 0
                                    
                                if status == 1:
                                    logger.info("Found status 1 in last message. Chat is complete.")
                                    await self.workflow_manager.finalize_conversation()
                                    return
                                else:
                                    latest_concept = await concepts_collection.find_one(
                                        {"enterprise_id": workflow_manager_enterprise_id_str},
                                        sort=[("ConceptCode", DESCENDING)]
                                    )
                                    CreationChatStatus = latest_concept.get('CreationChatStatus', 0)
                                    BluePrintReport = latest_concept.get('Blueprint', {})

                                    await self.websocket.send_text(json.dumps({
                                        "sender": last_agent.name,
                                        "content": last_message_content,
                                        "CreationChatStatus": CreationChatStatus,
                                        "BluePrintReport": BluePrintReport
                                    }))
                                    logger.info(f"Sent last message from {last_agent.name}: {last_message_content} with status: {CreationChatStatus}")
                                    last_message_sent = True
                                                            
                            except json.JSONDecodeError:
                                if last_agent.name != "Feedback_Agent":
                                    logger.error(f"Failed to parse last message as JSON: {last_message_content}")

                    else:
                        logger.info(f"Unexpected last speaker after resume: {last_agent.name if last_agent else 'None'}. Starting a new chat.")
                else:
                    logger.info(f"No saved state found. Starting a new chat for chat_id: {self.workflow_manager.chat_id}.")
            else:
                logger.info(f"No in-progress creation chat found for chat_id: {self.chat_id}. Starting a new chat.")

            if not resumed:
                # Update the CreationChatStatus to 0
                update_result = await concepts_collection.update_one(
                    {'_id': latest_concept['_id']},
                    {'$set': {'CreationChatStatus': 0}}
                )
                if update_result.modified_count > 0:
                    logger.info(f"Updated CreationChatStatus to 0 for chat_id: {self.chat_id}")
                else:
                    logger.warning(f"Failed to update CreationChatStatus for chat_id: {self.chat_id}")

                self.workflow_manager.initialize_new_chat()

                user_proxy = next(agent for agent in self.workflow_manager.agents if isinstance(agent, UserProxyWebAgent))

                # Use the VerificationChatState from the concept to build the initial message
                verification_state = latest_concept.get('VerificationChatState', {})
                agent_history = verification_state.get('AgentHistory', [])
                initial_message = f"""
                Please use the below User's Vision and Execute the Chat
                ---------------------------------------------------
                User's Vision: {agent_history}
                ---------------------------------------------------
                """
                user_proxy.iostream = IOWebsockets(self.websocket)
                user_proxy.workflow_manager = self.workflow_manager

                await user_proxy.a_initiate_chat(self.workflow_manager.groupchat_manager, initial_message)
                last_agent = self.workflow_manager.LastSpeaker
                last_message = self.workflow_manager.AgentHistory[-1] if self.workflow_manager.AgentHistory else None

            # Main chat loop
            while not chat_ended:
                try:
                    user_input = await self.websocket.receive_text()
                    user_input_json = {"content": user_input}
                    logger.info(f"User input received: {user_input_json['content']}")
                    
                    if not last_message_sent or user_input_json['content'] != last_message.get('content', ''):
                        chat_ended = await self.workflow_manager.groupchat_manager.a_receive(user_input_json['content'], self.workflow_manager.agents[0])
                    else:
                        last_message_sent = False
                    
                    logger.info(f"Chat ended status after processing: {chat_ended}")
                    
                    await self.workflow_manager.calculate_and_update_usage()

                    if user_input.lower() == "exit":
                        logger.info("User requested exit. Ending chat.")
                        chat_ended = True

                    last_agent = self.workflow_manager.LastSpeaker
                    last_message = self.workflow_manager.AgentHistory[-1] if self.workflow_manager.AgentHistory else None

                except WebSocketDisconnect:
                    logger.info(f"WebSocket disconnected for chat_id {self.chat_id}. Saving state and waiting for reconnection.")
                    await self.workflow_manager.save_chat_state()
                    break
                except RuntimeError as e:
                    if "WebSocket is not connected" in str(e):
                        logger.info(f"WebSocket disconnected for chat_id {self.chat_id}. Saving state and waiting for reconnection.")
                        await self.workflow_manager.save_chat_state()
                        break
                    else:
                        logger.error(f"Unexpected RuntimeError in chat loop: {str(e)}")
                        logger.error(traceback.format_exc())
                        break
                except Exception as e:
                    logger.error(f"Error in chat loop: {str(e)}")
                    logger.error(traceback.format_exc())
                    break

        except Exception as e:
            logger.error(f"Error in run_chat: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            if chat_ended:
                # Double check the last message for status and also check BluePrintReport
                try:
                    # Check last message
                    last_message_content = self.workflow_manager.AgentHistory[-1].get('content', '') if self.workflow_manager.AgentHistory else None
                    status_from_message = None
                    
                    if last_message_content:
                        try:
                            message_data = json.loads(last_message_content) if isinstance(last_message_content, str) else last_message_content
                            if isinstance(message_data, dict):
                                status_from_message = message_data.get('status')
                        except (json.JSONDecodeError, AttributeError):
                            pass
                    
                    # Also check BluePrintReport status
                    latest_concept = await concepts_collection.find_one(
                        {"enterprise_id": workflow_manager_enterprise_id_str},
                        sort=[("ConceptCode", DESCENDING)]
                    )
                    blueprint_status = None
                    if latest_concept and 'Blueprint' in latest_concept:
                        blueprint_status = latest_concept['Blueprint'].get('status')
                    
                    logger.info(f"Final status check - Message status: {status_from_message}, BluePrint status: {blueprint_status}")
                    
                    # If either status is 1, finalize the conversation
                    if status_from_message == 1 or blueprint_status == 1:
                        await self.workflow_manager.finalize_conversation()
                    else:
                        logger.info("Chat ended but no status 1 found. Keeping status as 0.")
                except Exception as e:
                    logger.error(f"Error in final status check: {str(e)}")
                    logger.error(traceback.format_exc())
            else:
                logger.info("Chat interrupted but not completed. Keeping status as 0.")
            
            await self.workflow_manager.handle_disconnection()
            await self.workflow_manager.save_chat_state()
            await self.workflow_manager.display_token_usage()
            logger.info(f"Chat session ended for {self.workflow_manager.chat_id}")

    async def send_chat_history(self, include_last=True):
        try:
            messages_to_send = self.workflow_manager.AgentHistory[1:]
            messages_to_send = [
                message for message in messages_to_send
                if message.get('name') in ['user_proxy', 'Feedback_Agent']
            ]

            CreationChatStatus = await self.workflow_manager.get_creation_chat_status()

            # Ensure enterprise_id is string for concepts_collection
            workflow_manager_enterprise_id_str = str(self.workflow_manager.enterprise_id)
            # Fetch the latest concept document to get the Blueprint
            latest_concept = await concepts_collection.find_one(
                {"enterprise_id": workflow_manager_enterprise_id_str, "ConceptCode": {"$exists": True}},  
                sort=[("ConceptCode", DESCENDING)]
            )

            BluePrintReport = latest_concept.get('Blueprint', {}) if latest_concept else {}

            if not include_last and messages_to_send:
                messages_to_send = messages_to_send[:-1]

            for message in messages_to_send:
                sender = message.get('name', 'unknown')
                content = message.get('content', '')

                # Try to parse content as JSON to check for status
                try:
                    if isinstance(content, str):
                        parsed_content = json.loads(content)
                        if isinstance(parsed_content, dict) and 'status' in parsed_content:
                            continue  # Skip messages with status field
                except json.JSONDecodeError:
                    pass  # Not JSON, continue normally

                await self.websocket.send_text(json.dumps({
                    "sender": sender,
                    "content": content,
                    "CreationChatStatus": CreationChatStatus,
                    "BluePrintReport": BluePrintReport
                }))
                logger.info(f"Sent historical message from {sender}: {content[:100]}...")

            logger.info("Finished sending filtered chat history")
        except Exception as e:
            logger.error(f"Error sending chat history: {str(e)}")

add_initialization_coroutine(cc_initialize)

async def cc_handler(websocket: WebSocket, chat_id: str, user_id: str, enterprise_id: str):
    global client, llm_config_collection

    try:
        if client is None:
            logger.error("Client is not initialized. Running cc_initialize...")
            await cc_initialize()
        
        if client is None:
            raise ValueError("Client is still not initialized after cc_initialize. Check the initialization process.")
        
        autogen_chat = AutoGenChat(
            chat_id=chat_id, 
            websocket=websocket, 
            llm_config = llm_config,  
            client=client,
            user_id=user_id,
            enterprise_id=enterprise_id  # Pass enterprise_id here
        )

        # Ensure enterprise_id is string for concepts_collection
        enterprise_id_str = str(enterprise_id)

        resumed = await autogen_chat.workflow_manager.resume_group_chat()
        if resumed:
            logger.info(f"Resuming existing chat for chat_id: {chat_id}")
        else:
            logger.info(f"Starting new chat for chat_id: {chat_id}")

        await autogen_chat.run_chat()

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for chat_id: {chat_id}, enterprise_id: {enterprise_id_str}")
        if 'autogen_chat' in locals() and hasattr(autogen_chat, 'workflow_manager'):
            await autogen_chat.workflow_manager.save_chat_state()
            await autogen_chat.workflow_manager.handle_disconnection()

    except Exception as e:
        logger.error(f"Error in CV handler for chat_id {chat_id}: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            await websocket.send_text(json.dumps({
                "sender": "system",
                "content": f"An error occurred: {str(e)}"
            }))
        except:
            pass
    finally:
        logger.info(f"WebSocket connection closed for chat_id: {chat_id}, enterprise_id: {enterprise_id_str}")
        if 'autogen_chat' in locals() and hasattr(autogen_chat, 'workflow_manager'):
            await autogen_chat.workflow_manager.save_chat_state()
            await autogen_chat.workflow_manager.handle_disconnection()

shared_websocket_manager.register_handler("cc", cc_handler)
