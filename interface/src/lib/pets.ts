export const PETS = [
  {
    id: "pomeranian",
    label: "Pomeranian",
    src: "/tinydog-avatar.png",
    alt: "Tiny orange Pomeranian companion",
    eyes: { left: [40.3, 43.2], right: [54.8, 36.8] },
  },
  {
    id: "french-bulldog",
    label: "French bulldog",
    src: "/pets/french-bulldog.png",
    alt: "Tiny French bulldog companion",
    eyes: { left: [40.0, 41.3], right: [60.1, 33.4] },
  },
  {
    id: "border-collie",
    label: "Border collie",
    src: "/pets/border-collie.png",
    alt: "Tiny border collie companion",
    eyes: { left: [43.2, 35.3], right: [57.5, 28.0] },
  },
  {
    id: "teddy-poodle",
    label: "Teddy poodle",
    src: "/pets/teddy-poodle.png",
    alt: "Tiny teddy-bear poodle companion",
    eyes: { left: [41.1, 37.9], right: [56.1, 32.0] },
  },
  {
    id: "samoyed",
    label: "Samoyed",
    src: "/pets/samoyed.png",
    alt: "Tiny white Samoyed companion",
    eyes: { left: [40.8, 38.6], right: [56.0, 33.2] },
  },
  {
    id: "golden-retriever",
    label: "Golden retriever",
    src: "/pets/golden-retriever.png",
    alt: "Tiny golden retriever companion",
    eyes: { left: [40.4, 30.3], right: [55.6, 24.1] },
  },
  {
    id: "labrador",
    label: "Labrador",
    src: "/pets/labrador.png",
    alt: "Tiny yellow Labrador companion",
    eyes: { left: [41.1, 33.1], right: [57.1, 26.6] },
  },
  {
    id: "british-blue-shorthair",
    label: "British blue",
    src: "/pets/british-blue-shorthair.png",
    alt: "Tiny British blue shorthair cat companion",
    eyes: { left: [41.1, 28.7], right: [56.4, 23.8] },
  },
  {
    id: "ragdoll",
    label: "Ragdoll",
    src: "/pets/ragdoll.png",
    alt: "Tiny ragdoll cat companion",
    eyes: { left: [48.8, 22.9], right: [60.3, 28.2] },
  },
  {
    id: "orange-tabby",
    label: "Orange tabby",
    src: "/pets/orange-tabby.png",
    alt: "Tiny original orange tabby cat companion",
    eyes: { left: [39.1, 24.8], right: [53.6, 29.9] },
  },
] as const;

export type PetId = (typeof PETS)[number]["id"];
export type PetOption = (typeof PETS)[number];

export const DEFAULT_PET_ID: PetId = "pomeranian";

export function isPetId(value: unknown): value is PetId {
  return typeof value === "string" && PETS.some((pet) => pet.id === value);
}

export function petById(id: PetId): PetOption {
  return PETS.find((pet) => pet.id === id) ?? PETS[0];
}

export function nextPetId(id: PetId): PetId {
  const currentIndex = PETS.findIndex((pet) => pet.id === id);
  return PETS[(currentIndex + 1) % PETS.length].id;
}
