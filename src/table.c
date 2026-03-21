#include <stdlib.h>
#include <string.h>

#include "tensorpy/common.h"
#include "tensorpy/object.h"
#include "tensorpy/table.h"
#include "tensorpy/value.h"

#define TABLE_MAX_LOAD 0.75

void initTable(Table* table) {
    table->count = 0;
    table->capacity = 0;
    table->entries = NULL;
}

void freeTable(Table* table) {
    free(table->entries);
    initTable(table);
}

// Simple identity hash for numbers and bit-shuffling for others
static uint32_t hashValue(Value value) {
    switch (value.type) {
        case VAL_BOOL:   return AS_BOOL(value) ? 1 : 0;
        case VAL_NIL:    return 0;
        case VAL_NUMBER: {
            double num = AS_NUMBER(value);
            uint32_t hash;
            memcpy(&hash, &num, sizeof(uint32_t)); // Simplified hashing
            return hash;
        }
        case VAL_OBJ: {
            if (IS_STRING(value)) {
                return AS_STRING(value)->hash;
            }
            if (IS_INT(value)) {
                return intHash(AS_INT(value));
            }
            // For other objects, use their memory address as a fallback
            return (uint32_t)(uintptr_t)AS_OBJ(value);
        }
    }
    return 0;
}


static Entry* findEntry(Entry* entries, int capacity, Value key) {
    uint32_t hash = hashValue(key);
    uint32_t index = hash % capacity;
    Entry* tombstone = NULL;

    for (;;) {
        Entry* entry = &entries[index];
        if (IS_NIL(entry->key)) {
            if (IS_NIL(entry->value)) {
                // Empty entry
                return tombstone != NULL ? tombstone : entry;
            } else {
                // We found a tombstone
                if (tombstone == NULL) tombstone = entry;
            }
        } else if (valuesEqual(entry->key, key)) {
            // We found the key
            return entry;
        }

        index = (index + 1) % capacity;
    }
}

static void adjustCapacity(Table* table, int capacity) {
    Entry* entries = (Entry*)malloc(sizeof(Entry) * capacity);
    for (int i = 0; i < capacity; i++) {
        entries[i].key = NIL_VAL;
        entries[i].value = NIL_VAL;
    }

    table->count = 0;
    for (int i = 0; i < table->capacity; i++) {
        Entry* entry = &table->entries[i];
        if (IS_NIL(entry->key)) continue;

        Entry* dest = findEntry(entries, capacity, entry->key);
        dest->key = entry->key;
        dest->value = entry->value;
        table->count++;
    }

    if (table->entries != NULL) {
        free(table->entries);
    }
    table->entries = entries;
    table->capacity = capacity;
}

bool tableGet(Table* table, Value key, Value* value) {
    if (table->count == 0) return false;

    Entry* entry = findEntry(table->entries, table->capacity, key);
    if (IS_NIL(entry->key)) return false;

    *value = entry->value;
    return true;
}

bool tableSet(Table* table, Value key, Value value) {
    if (table->count + 1 > table->capacity * TABLE_MAX_LOAD) {
        int capacity = table->capacity < 8 ? 8 : table->capacity * 2;
        adjustCapacity(table, capacity);
    }

    Entry* entry = findEntry(table->entries, table->capacity, key);
    bool isNewKey = IS_NIL(entry->key);
    if (isNewKey && IS_NIL(entry->value)) table->count++;

    entry->key = key;
    entry->value = value;
    return isNewKey;
}

bool tableDelete(Table* table, Value key) {
    if (table->count == 0) return false;

    // Find the entry
    Entry* entry = findEntry(table->entries, table->capacity, key);
    if (IS_NIL(entry->key)) return false;

    // Place a tombstone in the entry
    entry->key = NIL_VAL;
    entry->value = BOOL_VAL(true);
    return true;
}

ObjString* tableFindString(Table* table, const char* chars, int length, uint32_t hash) {
    if (table->count == 0) return NULL;

    uint32_t index = hash % table->capacity;
    for (;;) {
        Entry* entry = &table->entries[index];
        if (IS_NIL(entry->key)) {
            // Stop if we find an empty non-tombstone entry
            if (IS_NIL(entry->value)) return NULL;
        } else if (IS_STRING(entry->key)) {
            ObjString* string = AS_STRING(entry->key);
            if (string->length == length &&
                string->hash == hash &&
                memcmp(string->chars, chars, length) == 0) {
                return string;
            }
        }

        index = (index + 1) % table->capacity;
    }
}

void tableAddAll(Table* from, Table* to) {
    for (int i = 0; i < from->capacity; i++) {
        Entry* entry = &from->entries[i];
        if (!IS_NIL(entry->key)) {
            tableSet(to, entry->key, entry->value);
        }
    }
}

void tableRemoveWhite(Table* table) {
    for (int i = 0; i < table->capacity; i++) {
        Entry* entry = &table->entries[i];
        if (IS_NIL(entry->key)) {
            continue;
        }
        if (IS_OBJ(entry->key) && !AS_OBJ(entry->key)->isMarked) {
            tableDelete(table, entry->key);
        }
    }
}
