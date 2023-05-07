import argparse
import datetime
import json
import os
import os.path
import re
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import colorama
import pytz
from colorama import Fore, Style
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langchain import LLMChain, PromptTemplate
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    AgentType,
    Tool,
    ZeroShotAgent,
    initialize_agent,
)
from langchain.agents.utils import validate_tools_single_input
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.memory import (
    ConversationBufferMemory,
    ConversationBufferWindowMemory,
    ConversationEntityMemory,
)
from langchain.prompts import BaseChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain.schema import (
    AgentAction,
    AgentFinish,
    AIMessage,
    BaseMessage,
    BaseOutputParser,
    HumanMessage,
    SystemMessage,
)
from langchain.tools.base import BaseTool
from langchain.utilities import SerpAPIWrapper

from prompt import (
    FORMAT_INSTRUCTIONS,
    SCHEDULE_BOT_PREFIX,
    SCHEDULE_BOT_SUFFIX,
    TEMPLATE_TOOL_RESPONSE,
    USER_CONTEXT,
)

colorama.init(autoreset=True)

VERBOSE_FLAG = False
## Load models
chat4 = ChatOpenAI(
    temperature=0,
    model_name="gpt-4",
    streaming=VERBOSE_FLAG,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=VERBOSE_FLAG,
)  # type: ignore
chat35 = ChatOpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    streaming=VERBOSE_FLAG,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=VERBOSE_FLAG,
)  # type: ignore
CHAT_MODEL = chat35


class IDMapping:
    """
    This class is used to map id1 to id2 and vice versa.
    id1 is simple id; id2 is complex (real) id.
    id1 and id2 are one-to-one mapping.
    """

    counter = 1
    id1_to_id2 = {}
    id2_to_id1 = {}

    def add(self, id2):
        if id2 in self.id2_to_id1:
            return
        self.id1_to_id2[self.counter] = id2
        self.id2_to_id1[id2] = self.counter
        self.counter += 1

    def get_id1(self, id2):
        return self.id2_to_id1[id2]

    def get_id2(self, id1):
        return self.id1_to_id2[id1]


idMaps = IDMapping()


class LLMFormatter:
    """
    This class is used to format user input to a required format using LLM.
    """

    llm: BaseChatModel
    system_message_prompt = SystemMessagePromptTemplate.from_template(
        "You are a helpful assistant that format user inputs. You will be give a context, a user input, and a required outpout format. You output should follow exactly the output format. Preserve values in user input, only fill in vlues from context when missing from user input. The required format is: {format}. The context is: {context}. "
    )

    human_message_prompt = HumanMessagePromptTemplate.from_template("{user_input}")

    def __init__(self, llm):
        self.llm = llm

    def format(
        self, format: str, user_input: str, context: str = "", example: str = ""
    ):
        chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.human_message_prompt]
        )
        messages = chat_prompt.format_prompt(
            format=format,
            user_input=user_input,
            context=context,
        ).to_messages()
        # print("Formatter: ", messages)
        return self.llm(messages).content


llmFormatter = LLMFormatter(CHAT_MODEL)
# llmFormatter = LLMFormatter(chat35)


class GoogleCalendar:
    ## Get Google Calendar Evetns
    # If modifying these scopes, delete the file token.json.
    SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

    system_calendar = ""
    today_events = ""
    creds = None

    def __init__(self) -> None:
        self._auth_google_calendar()
        self.today_events = self.get_today_events()
        self.system_calendar = self.construct_system_calendar()

    def _auth_google_calendar(self):
        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists("token.json"):
            self.creds = Credentials.from_authorized_user_file(
                "token.json", self.SCOPES
            )
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", self.SCOPES
                )
                self.creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(self.creds.to_json())

    def get_today_events(self, query=""):
        # Get the current date and time
        now = datetime.datetime.now(pytz.timezone("US/Eastern"))

        # Get the last second of today
        end_of_today = datetime.datetime(
            now.year, now.month, now.day, 23, 59, 59, tzinfo=now.tzinfo
        )

        try:
            service = build("calendar", "v3", credentials=self.creds)
            # Call the Calendar API
            now = datetime.datetime.utcnow().isoformat() + "Z"  # 'Z' indicates UTC time
            events_result = (
                service.events()
                .list(
                    calendarId="primary",
                    timeMin=now,
                    maxResults=10,
                    singleEvents=True,
                    orderBy="startTime",
                    timeMax=end_of_today.isoformat(),
                )
                .execute()
            )

            events = events_result.get("items", [])

            if not events:
                print("No upcoming events found.")
                return []
            else:
                events_LLM = [self.extract_cal_event(event) for event in events]
                return events_LLM

        except HttpError as error:
            print("An error occurred: %s" % error)

    def extract_cal_event(self, event):
        # Selected Keys in calendar events for LLM
        calendar_keys_LLM = {"id", "summary", "start", "end"}
        result = {k: event[k] for k in calendar_keys_LLM}
        result["start"] = result["start"].get("date", result["start"].get("dateTime"))
        result["end"] = result["end"].get("date", result["end"].get("dateTime"))

        # only add new evetns to idMaps
        if result["id"] not in idMaps.id2_to_id1:
            idMaps.add(result["id"])
        result["id"] = idMaps.get_id1(result["id"])
        return result

    def construct_system_calendar(self, qeury=""):
        today_events_str = ""

        if not self.today_events:
            return "No event"
        else:
            return ",".join([str(event) for event in self.today_events])

    def quickAdd(self, query: str) -> str:
        try:
            service = build("calendar", "v3", credentials=self.creds)

            # Call the Calendar API
            created_event = (
                service.events().quickAdd(calendarId="primary", text=query).execute()
            )
            self.update_system_calendar()
            return "Event with id %s added " % created_event["id"]

        except HttpError as error:
            warnings.warn("An error occurred: %s" % error)
            return "An error occurred: %s" % error

    def insert_event(self, query: str) -> str:
        # Insert an event.
        # Query should be comma separated string of event name, start time, end time
        try:
            service = build("calendar", "v3", credentials=self.creds)
            try:
                summary, start, end = [element.strip() for element in query.split(",")]
            except ValueError:
                formatter_input = llmFormatter.format(
                    "summary,start_time,end_time", query, "", ""
                )
                summary, start, end = [
                    element.strip() for element in formatter_input.split(",")
                ]
            # Call the Calendar API
            event = {
                "summary": summary,
                "start": {"dateTime": start, "timeZone": "America/New_York"},
                "end": {"dateTime": end, "timeZone": "America/New_York"},
            }

            created_event = (
                service.events().insert(calendarId="primary", body=event).execute()
            )
            self.update_system_calendar()
            return "Event with id %s added: " % idMaps.get_id1(
                created_event["id"]
            ) + " ".join([summary, start, end])

        except HttpError as error:
            warnings.warn("An error occurred: %s" % error)
            return "An error occurred: %s" % error

    def delete_event(self, event_id: str) -> None:
        try:
            service = build("calendar", "v3", credentials=self.creds)

            # Call the Calendar API
            service.events().delete(
                calendarId="primary", eventId=idMaps.get_id2(int(event_id))
            ).execute()
            self.update_system_calendar()
        except HttpError as error:
            print("An error occurred: %s" % error)

    def update_event(self, query: str) -> None:
        try:
            service = build("calendar", "v3", credentials=self.creds)
            try:
                event_id, summary, start, end = [
                    element.strip() for element in query.split(",")
                ]
            except ValueError:
                formatter_input = llmFormatter.format(
                    "id,summary,start_time,end_time", query, self.system_calendar, ""
                )
                event_id, summary, start, end = [
                    element.strip() for element in formatter_input.split(",")
                ]
            # Call the Calendar API
            event = (
                service.events()
                .get(calendarId="primary", eventId=idMaps.get_id2(int(event_id)))
                .execute()
            )
            event["summary"] = summary
            event["start"]["dateTime"] = start
            event["end"]["dateTime"] = end
            updated_event = (
                service.events()
                .update(
                    calendarId="primary",
                    eventId=idMaps.get_id2(int(event_id)),
                    body=event,
                )
                .execute()
            )
            self.update_system_calendar()
            return updated_event

        except HttpError as error:
            print("An error occurred: %s" % error)

    def update_system_calendar(self) -> None:
        self.today_events = self.get_today_events()
        self.system_calendar = self.construct_system_calendar()


# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format_messages(self, **kwargs) -> List[BaseMessage]:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]


class ScheduleAgent(ZeroShotAgent):
    """
    Class that represents the schedule agent.
    """

    @classmethod
    def create_prompt(
        cls,
        tools: Sequence[BaseTool],
        system_message: str = SCHEDULE_BOT_PREFIX,
        human_message: str = SCHEDULE_BOT_SUFFIX,
        input_variables: Optional[List[str]] = None,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> BasePromptTemplate:
        tool_strings = "\n".join(
            [f"> {tool.name}: {tool.description}" for tool in tools]
        )
        tool_names = ", ".join([tool.name for tool in tools])
        _output_parser = output_parser or cls._get_default_output_parser()
        format_instructions = human_message.format(
            format_instructions=_output_parser.get_format_instructions()
        )

        final_prompt = format_instructions.format(
            tool_names=tool_names, tools=tool_strings
        )
        if input_variables is None:
            input_variables = [
                "input",
                "chat_history",
                "agent_scratchpad",
                "user_context",
                "system_calendar",
                "current_time",
            ]
        messages = [
            SystemMessagePromptTemplate.from_template(system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(final_prompt),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
        return ChatPromptTemplate(input_variables=input_variables, messages=messages)

    def _construct_scratchpad(
        self, intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> List[BaseMessage]:
        """Construct the scratchpad that lets the agent continue its thought process."""
        thoughts: List[BaseMessage] = []
        for action, observation in intermediate_steps:
            thoughts.append(AIMessage(content=action.log))
            human_message = HumanMessage(
                content=TEMPLATE_TOOL_RESPONSE.format(observation=observation)
            )
            thoughts.append(human_message)
        return thoughts

    @classmethod
    def from_llm_and_tools(
        cls,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        output_parser: Optional[AgentOutputParser] = None,
        **kwargs: Any,
    ) -> ZeroShotAgent:
        """
        Create an agent from a language model and a list of tools.
        """
        cls._validate_tools(tools)
        prompt = cls.create_prompt(tools=tools, output_parser=output_parser)

        llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=VERBOSE_FLAG)
        tool_names = [tool.name for tool in tools]
        _output_parser = output_parser or cls._get_default_output_parser()

        return cls(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            output_parser=_output_parser,
            **kwargs,
        )

    @classmethod
    def _validate_tools(cls, tools: Sequence[BaseTool]) -> None:
        validate_tools_single_input(cls.__name__, tools)
        for tool in tools:
            if tool.description is None:
                raise ValueError(
                    f"Got a tool {tool.name} without a description. For this agent, "
                    f"a description must always be provided."
                )
        super()._validate_tools(tools)


class CustomOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # print("=============")
        # print("Raw LLM Output:")
        # print(llm_output)
        # print("=============")
        # Check if agent should finish
        if "Final Answer" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        # regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        regex = r'"action"\s*:\s*"([^"]+)"\s*,\s*"action_input"\s*:\s*"([^"]*)"'
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            # raise ValueError(f"Could not parse LLM output: `{llm_output}`")
            warnings.warn(f"Could not parse LLM output: `{llm_output}`")
            return AgentFinish(
                return_values={"output": "[WARNING: Raw Ouptut]: " + llm_output},
                log=llm_output,
            )
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


def parse_executor_output(output: str) -> str:
    """
    Parse the output of the executor.
    """
    pattern = r'"action_input":\s*"(.+?)"'

    if "action" in output and "Final Answer" in output and "action_input" in output:
        match = re.search(pattern, output)
        if match:
            action_input = match.group(1)
            return action_input

    return "WARNING Output not parsed\n" + output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", help="increase output verbosity", action="store_true"
    )
    parser.add_argument("--chat4", help="using chat4", action="store_true")
    args = parser.parse_args()
    if args.verbose:
        VERBOSE_FLAG = True
    if args.chat4:
        CHAT_MODEL = chat4
    calendar = GoogleCalendar()

    tools = [
        Tool(
            name="Add Event",
            func=calendar.insert_event,
            description="A command to create an event. Action input should be a comma seperated list of 3 arguemtns representing the event: event sumary, start time, end time. Example usage: Go to gym,2023-01-01T09:00:00,2023-01-01T10:00:00",
        ),
        Tool(
            name="Delete Event",
            func=calendar.delete_event,
            description="A command to delete an event from the calendar. Action input is just the event id. Example usage: 3",
        ),
        Tool(
            name="Update Event",
            func=calendar.update_event,
            description="A command to update an event from the calendar. The input to this tool should be a comma separated list of 4 arguments representing the new event with updated details: event id, summary, start time, and end time. Example usage: 4, Get a hair cut, 2015-05-28T09:00:00, 2015-05-28T10:00:00. This will update the event with id 4 to have the new details.",
        ),
        Tool(
            name="Get Today Calendar",
            func=calendar.construct_system_calendar,
            description="A command to get today's calendar. Action input should be empty string.",
        ),
    ]

    memory = ConversationBufferMemory(
        memory_key="chat_history", input_key="input", return_messages=True
    )
    scheduleAgent = ScheduleAgent.from_llm_and_tools(
        tools=tools, llm=CHAT_MODEL, output_parser=CustomOutputParser()
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=scheduleAgent, tools=tools, verbose=VERBOSE_FLAG, memory=memory
    )
    # print(scheduleAgent.llm_chain.prompt)

    # write a loop that takes in user input and generate system response with chatgpt
    while True:
        # user_input = "What's on my calendar?"
        user_input = input(Fore.CYAN + "Enter your message: " + Style.RESET_ALL)
        agent_output = agent_executor.run(
            input=user_input,
            system_calendar=calendar.system_calendar,
            user_context=USER_CONTEXT,
            current_time=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        )
        print(Fore.MAGENTA + "Agent: ", parse_executor_output(agent_output))
