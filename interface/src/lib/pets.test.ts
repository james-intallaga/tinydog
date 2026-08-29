import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_PET_ID, isPetId, nextPetId, petById, PETS } from "./pets";

test("the companion roster contains the pom and nine requested alternatives", () => {
  assert.equal(PETS.length, 10);
  assert.equal(new Set(PETS.map((pet) => pet.id)).size, PETS.length);
  assert.deepEqual(
    PETS.map((pet) => pet.label),
    [
      "Pomeranian",
      "French bulldog",
      "Border collie",
      "Teddy poodle",
      "Samoyed",
      "Golden retriever",
      "Labrador",
      "British blue",
      "Ragdoll",
      "Orange tabby",
    ],
  );
});

test("pet cycling advances in roster order and wraps around", () => {
  assert.equal(nextPetId("pomeranian"), "french-bulldog");
  assert.equal(nextPetId("orange-tabby"), "pomeranian");
});

test("only declared pet ids can be restored from local storage", () => {
  assert.equal(DEFAULT_PET_ID, "pomeranian");
  assert.equal(isPetId("ragdoll"), true);
  assert.equal(isPetId("unknown-pet"), false);
  assert.equal(isPetId(null), false);
  assert.equal(petById("ragdoll").src, "/pets/ragdoll.png");
});
