import {
    useStore_MessageLogs,
    useStore_MessageInputValue,
    store,
} from "@store";

import { useStdoutToPython } from "@useStdoutToPython";

const COOLDOWN = 2000; // 2 seconds

export const useMessage = () => {
    const { currentMessageLogs, addMessageLogs, updateMessageLogs } = useStore_MessageLogs();
    const { currentMessageInputValue, updateMessageInputValue } = useStore_MessageInputValue();
    const { asyncStdoutToPython } = useStdoutToPython();

    const sendMessage = (message) => {
        const uuid = crypto.randomUUID();
        const send_message_object = {
            id: uuid,
            message: message,
        };
        asyncStdoutToPython("/run/send_message_box", send_message_object);

        addMessageLogs({
            id: uuid,
            category: "sent",
            status: "pending",
            created_at: generateTimeData(),
            messages: {
                original: { message: message, transliteration: [] },
                translations: [],
            },
        });
    };

    const addSystemMessageLog = (message) => {
        const uuid = crypto.randomUUID();
        const date = generateTimeData();

        addMessageLogs({
            id: uuid,
            category: "system",
            status: "system",
            created_at: date,
            messages: {
                original: { message: message, transliteration: [] },
                translations: [],
            },
        });
    };

    const addSystemMessageLog_FromBackend = (payload) => {
        addSystemMessageLog(payload.message);
    };

    const updateSentMessageLogById = (payload) => {
        updateMessageLogs(updateItemById(payload.id, payload));
    };

    const addSentMessageLog = (payload) => {
        const message_object = generateMessageObject(payload, "sent");
        updateMessageLogs(upsertMessageObject(message_object));
    };

    const addReceivedMessageLog = (payload) => {
        const message_object = generateMessageObject(payload, "received");
        updateMessageLogs(upsertMessageObject(message_object));
    };

    const startTyping = () => {
        const now = Date.now();
        if (now - store.last_executed_time_startTyping >= 2000) {
            store.last_executed_time_startTyping = now;
            asyncStdoutToPython("/run/typing_message_box");
        }
    };

    const stopTyping = () => {
        asyncStdoutToPython("/run/stop_typing_message_box");
    };

    return {
        currentMessageLogs,
        sendMessage,
        addSystemMessageLog,
        addSystemMessageLog_FromBackend,
        updateSentMessageLogById,
        addSentMessageLog,
        addReceivedMessageLog,

        currentMessageInputValue,
        updateMessageInputValue,

        startTyping,
        stopTyping,
    };
};

const generateTimeData = () => {
    return new Date().toLocaleTimeString(
        "ja-JP",
        { hour12: false, hour: "2-digit", minute: "2-digit" }
    );
};

const generateMessageObject = (data, category, status = "ok") => {
    return {
        id: data.id ?? crypto.randomUUID(),
        created_at: generateTimeData(),
        category: category,
        status: status,
        source: data.source,
        messages: {
            original: data.original,
            translations: data.translations ?? [],
        },
    };
};

// Transcription keeps sending longer versions of the utterance that is still
// being spoken, all with the same id. Replace the entry in place (keeping its
// position and time) instead of adding a new one for every version.
const upsertMessageObject = (message_object) => (current_items) => {
    const index = current_items.data.findIndex((item) => item.id === message_object.id);
    if (index === -1) return [...current_items.data, message_object];
    const next = [...current_items.data];
    next[index] = { ...message_object, created_at: next[index].created_at };
    return next;
};

const updateItemById = (id, updated_data) => (current_items) => {
    return current_items.data.map(item => {
        if (item.id === id) {
            if (updated_data.messages) {
                return { ...item, ...updated_data, created_at: item.created_at };
            }
            return {
                ...item,
                status: "ok",
                messages: {
                    ...item.messages,
                    translations: updated_data.translations ?? item.messages.translations,
                },
            };
        }
        return item;
    });
};